# -*- coding: utf-8 -*-
#modeling_ir.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from modeling_bert import BertModel
from typing import Dict, Optional, cast, List

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.modules import Seq2SeqEncoder, ConditionalRandomField
from allennlp.training.metrics import CategoricalAccuracy
import math

class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class IntentionLabelTagger(Model):
    """
    在forward函数中加入了对抗学习和R-Drop的功能,通过adv_alpha和r_drop_alpha两个参数控制是否启用以及loss的权重
    对抗学习部分,在utterance encoder的输出上叠加一个随机扰动,然后重新计算后续的输出,并在两个输出间计算KL divergence作为loss
    R-Drop部分,同时前向计算两次,得到两组输出logits,分别计算它们之间的KL divergence,取平均作为loss,促进一致性
    将CRF loss、对抗loss和R-Drop loss相加作为总的loss
    """
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_load_path: str,
        dialogue_encoder: Seq2SeqEncoder,
        dropout: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        adv_alpha: float = 1.0,
        r_drop_alpha: float = 1.0,
        # action_weights: Optional[torch.Tensor] = None,
        # intent_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.utterance_encoder = BertModel.from_pretrained(transformer_load_path)
        self.dialogue_encoder = dialogue_encoder
        self.act_decoder = ClassificationHead(input_dim=self.dialogue_encoder.get_output_dim(),
                                              inner_dim=self.dialogue_encoder.get_output_dim(),
                                              num_classes=self.vocab.get_vocab_size('action_labels'),
                                              pooler_dropout=0.3)
        self.intent_decoder = ClassificationHead(input_dim=self.dialogue_encoder.get_output_dim(),
                                                 inner_dim=self.dialogue_encoder.get_output_dim(),
                                                 num_classes=self.vocab.get_vocab_size('intention_labels'),
                                                 pooler_dropout=0.3)
        ## 
        self.speaker_embeds = self.utterance_encoder.embeddings.speaker_embeddings
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.calculate_accuracy_act = CategoricalAccuracy()
        self.calculate_accuracy_int = CategoricalAccuracy()
        self.crf_act = ConditionalRandomField(self.vocab.get_vocab_size('action_labels'))
        self.crf_int = ConditionalRandomField(self.vocab.get_vocab_size('intention_labels'))
        # 对抗学习
        self.adv_alpha = adv_alpha
        self.r_drop_alpha = r_drop_alpha
        initializer(self)
        self.position_embeddings = nn.Embedding(512, 32)  # 位置编码的embedding层
        self.dialogue_encoder.input_size = 768 + 32
        # self.action_weights = action_weights # 新增
        # self.intent_weights = intent_weights # 新增


    @overrides
    def forward(self, dialogue, speaker, intentions = None, actions = None, **kwargs,):
        # 对抗学习中adv_alpha、r_drop_alpha参数
        adv_alpha = self.adv_alpha
        r_drop_alpha = self.r_drop_alpha
        batch_size, utter_len, seq_len = dialogue.shape
        dialogue = dialogue.reshape(batch_size * utter_len, seq_len)
        '''
        utterance feature
        '''
        speaker_ids = torch.repeat_interleave(speaker, seq_len, -1)
        speaker_ids = speaker_ids.reshape(batch_size * utter_len, seq_len)
        encoded_utterance = self.utterance_encoder(input_ids=dialogue,
                                                   attention_mask=dialogue != 0,
                                                   speaker_ids=torch.clamp(speaker_ids,min=0),
                                                   use_cache=True,
                                                   return_dict=True)['last_hidden_state']
        encoded_utterance = encoded_utterance.reshape(batch_size, utter_len, seq_len, -1)
        encoded_utterance = encoded_utterance[:,:,0,:]
        encoded_utterance = self.dropout(encoded_utterance) if self.dropout else encoded_utterance
        
        '''
        dialogue feature
        '''
        ##
        speaker_embeds = self.speaker_embeds(speaker)
        encoded_utterance = encoded_utterance + speaker_embeds
        encoded_utterance = encoded_utterance 

        # encoded_dialogue = self.dialogue_encoder(encoded_utterance, None)
        # encoded_dialogue = self.dropout(encoded_dialogue) if self.dropout else encoded_dialogue
        # # [batch_size, utterance_number, utterance_embedding_size]


        # 生成句子的相对位置编码
        batch_size, utter_len, hidden_size = encoded_utterance.shape
        position_ids = torch.arange(utter_len, dtype=torch.long, device=encoded_utterance.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, utter_len)
        position_embeddings = self.position_embeddings(position_ids)  # [batch_size, utter_len, position_emb_dim]
        # 将位置编码与句子编码concat
        encoded_utterance = torch.cat([encoded_utterance, position_embeddings], dim=-1)
        # [batch_size, utter_len, hidden_size+position_emb_dim]

        encoded_dialogue = self.dialogue_encoder(encoded_utterance, None)
        encoded_dialogue = self.dropout(encoded_dialogue) if self.dropout else encoded_dialogue

        '''
        对抗学习:在utterance encoder的输出上加扰动
        '''
        if self.training and adv_alpha > 0:
            # 生成随机扰动
            adv_noise = torch.randn(encoded_utterance.shape).to(encoded_utterance.device) 
            # 在utterance encoder上叠加扰动
            encoded_utterance_adv = encoded_utterance + adv_alpha * adv_noise
            encoded_dialogue_adv = self.dialogue_encoder(encoded_utterance_adv, None)

        '''
        R-Drop:同时前向计算两次,在输出层加consistency loss
        '''
        if self.training and r_drop_alpha > 0:
            # 再次前向计算一次
            encoded_dialogue_r = self.dialogue_encoder(encoded_utterance, None) 

        '''
        decoder
        '''
        encoded_dialogue_act = self.act_decoder(encoded_dialogue)
        encoded_dialogue_int = self.intent_decoder(encoded_dialogue)

        if self.training and adv_alpha > 0:
            # 对抗学习的decoder输出
            encoded_dialogue_act_adv = self.act_decoder(encoded_dialogue_adv)
            encoded_dialogue_int_adv = self.intent_decoder(encoded_dialogue_adv)
        if self.training and r_drop_alpha > 0:  
            # R-Drop的decoder输出
            encoded_dialogue_act_r = self.act_decoder(encoded_dialogue_r)
            encoded_dialogue_int_r = self.intent_decoder(encoded_dialogue_r)
        
        '''
        metric
        '''
        output = dict()

        '''
        actions metric
        '''  
        labels_mask = speaker != -1
        best_paths_act = self.crf_act.viterbi_tags(encoded_dialogue_act, labels_mask, top_k=1)
        predicted_acts = cast(List[List[int]], [x[0][0] for x in best_paths_act])
        output['actions'] = predicted_acts
        if actions is not None:
            self.calculate_accuracy_act(encoded_dialogue_act, actions, labels_mask)

        '''
        intentions metric
        '''
        best_paths_int = self.crf_int.viterbi_tags(encoded_dialogue_int, labels_mask, top_k=1) 
        predicted_ints = cast(List[List[int]], [x[0][0] for x in best_paths_int])
        output['intentions'] = predicted_ints
        if intentions is not None:
            self.calculate_accuracy_int(encoded_dialogue_int, intentions, labels_mask)

        '''
        loss
        '''

        if actions != None and intentions != None:
            # if self.training:
            #     log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
            #     log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
            #     crf_loss = (-log_likelihood_act) + (-log_likelihood_int)
            #     output["loss"] = crf_loss


                # log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
                # log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
                
                # fl_gamma = 2.0
                # eps = 1e-8  # 添加一个很小的正数,避免log为0导致错误
                # neg_act_logits = -encoded_dialogue_act

                # # act focal loss
                # neg_act_probs = neg_act_logits.exp() # 计算负概率
                # act_probs = (1 - neg_act_probs + eps).clamp(min=eps) # 避免为0
                # act_log_probs = act_probs.log()  
                # act_loss = act_log_probs * (1 - act_probs).pow(fl_gamma)
                # act_loss = act_loss[range(len(actions)), actions]
                # fl_act = -act_loss.mean()

                # # intention focal loss  
                # neg_int_logits = -encoded_dialogue_int
                # neg_int_probs = neg_int_logits.exp()
                # int_probs = (1 - neg_int_probs + eps).clamp(min=eps)
                # int_log_probs = int_probs.log()
                # int_loss = int_log_probs * (1 - int_probs).pow(fl_gamma)  
                # int_loss = int_loss[range(len(intentions)), intentions]
                # fl_int = -int_loss.mean()
                # output["loss"] = fl_act + fl_int + (-log_likelihood_act) + (-log_likelihood_int)


                # log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
                # log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
                # # 对不同类别的loss进行加权
                # crf_loss_act = (-log_likelihood_act * self.action_weights[actions.view(-1)]).mean() 
                # crf_loss_int = (-log_likelihood_int * self.intent_weights[intentions.view(-1)]).mean()
                # crf_loss = crf_loss_act + crf_loss_int
                # output["loss"] = crf_loss


            # 不加权
            # else:
            log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
            log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
            crf_loss = (-log_likelihood_act) + (-log_likelihood_int)
            output["loss"] = crf_loss

            #     # method1:
            #     log_likelihood_act = self.crf_act(encoded_dialogue_act, actions, labels_mask)
            #     log_likelihood_int = self.crf_int(encoded_dialogue_int, intentions, labels_mask)
            #     # 对不同类别的loss进行加权
            #     crf_loss_act = (-log_likelihood_act * action_weights[actions.view(-1)]).mean()
            #     crf_loss_int = (-log_likelihood_int * intent_weights[intentions.view(-1)]).mean()
            #     crf_loss = crf_loss_act + crf_loss_int
            #     output["loss"] = crf_loss
            
            # method2:
            # focal loss for actions
            # logits_act = encoded_dialogue_act
            # probs_act = torch.softmax(logits_act, dim=-1)
            # log_probs_act = torch.log_softmax(logits_act, dim=-1)
            # focal_loss_act = torch.sum(- action_weights[actions.view(-1)] * (1 - probs_act.gather(2, actions.unsqueeze(2)).squeeze(2)) ** 2 * log_probs_act.gather(2, actions.unsqueeze(2)).squeeze(2), dim=-1)
            # focal_loss_act = focal_loss_act.mean()

            # # focal loss for intents
            # logits_int = encoded_dialogue_int
            # probs_int = torch.softmax(logits_int, dim=-1) 
            # log_probs_int = torch.log_softmax(logits_int, dim=-1)
            # focal_loss_int = torch.sum(- intent_weights[intentions.view(-1)] * (1 - probs_int.gather(2, intentions.unsqueeze(2)).squeeze(2)) ** 2 * log_probs_int.gather(2, intentions.unsqueeze(2)).squeeze(2), dim=-1)
            # focal_loss_int = focal_loss_int.mean()

            # focal_loss = focal_loss_act + focal_loss_int
            # output["loss"] = focal_loss

            '''
            对抗loss
            '''
            if self.training and adv_alpha > 0:
                adv_loss_act = F.kl_div(F.log_softmax(encoded_dialogue_act_adv, 2), 
                                        F.softmax(encoded_dialogue_act.detach(), 2), 
                                        reduction='batchmean')
                adv_loss_int = F.kl_div(F.log_softmax(encoded_dialogue_int_adv, 2),
                                        F.softmax(encoded_dialogue_int.detach(), 2),
                                        reduction='batchmean') 
                adv_loss = adv_loss_act + adv_loss_int
                output["loss"] += adv_alpha * adv_loss

            '''
            R-Drop loss
            '''
            if self.training and r_drop_alpha > 0:
                kl_loss_act = F.kl_div(F.log_softmax(encoded_dialogue_act, 2), 
                                    F.softmax(encoded_dialogue_act_r.detach(), 2),
                                    reduction='batchmean')
                kl_loss_int = F.kl_div(F.log_softmax(encoded_dialogue_int, 2),
                                    F.softmax(encoded_dialogue_int_r.detach(), 2), 
                                    reduction='batchmean')
                kl_loss = kl_loss_act + kl_loss_int
                # 交换下位置再算一次
                rkl_loss_act = F.kl_div(F.log_softmax(encoded_dialogue_act_r, 2),
                                        F.softmax(encoded_dialogue_act.detach(), 2),
                                        reduction='batchmean')
                rkl_loss_int = F.kl_div(F.log_softmax(encoded_dialogue_int_r, 2),
                                        F.softmax(encoded_dialogue_int.detach(), 2),
                                        reduction='batchmean')
                rkl_loss = rkl_loss_act + rkl_loss_int

                r_drop_loss = (kl_loss + rkl_loss) / 2
                output["loss"] += r_drop_alpha * r_drop_loss
    
        return output    


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = dict()
        
        act_accuracy = self.calculate_accuracy_act.get_metric(reset)
        int_accuracy = self.calculate_accuracy_int.get_metric(reset)
        
        metrics_to_return['action_accuracy'] = act_accuracy
        metrics_to_return['intent_accuracy'] = int_accuracy
        metrics_to_return['avg_accuracy'] = (act_accuracy + int_accuracy) / 2
        
        return metrics_to_return
    
