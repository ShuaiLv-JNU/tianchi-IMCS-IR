# -*- coding: utf-8 -*-
# train.py
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
prepare_environment(Params({"random_seed":1000, "numpy_seed":2000, "pytorch_seed":3000}))

import os
import torch
import logging
import argparse
import random
from sklearn.model_selection import KFold
import json

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models.model import Model
from allennlp.data import DataLoader, Vocabulary
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from allennlp.training.learning_rate_schedulers.linear_with_warmup import LinearWithWarmup

from transformers.optimization import AdamW

from modeling_ir import IntentionLabelTagger
from data_loader_ir import IntentionRecognitionDatasetReader

from typing import Dict, Any
from allennlp.data import Instance
from allennlp.data.fields import ListField, LabelField, TensorField

from collections import Counter


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
"""
构建instance类的词表
"""
def build_vocab(instances):
    return Vocabulary.from_instances(instances)
"""
通过IntentionLabelTagger构建模型
"""
def build_model(vocab: Vocabulary,
                transformer_load_path: str, 
                pretrained_hidden_size: int,
                # action_weights, intent_weights,
                adv_alpha: float,
                r_drop_alpha: float) -> Model:
    lstmencoder = LstmSeq2SeqEncoder(input_size=pretrained_hidden_size + 32, # 加了位置编码维度
                                     hidden_size=128,
                                     num_layers=1,
                                     bidirectional=True)
    return IntentionLabelTagger(vocab=vocab,
                                dialogue_encoder=lstmencoder,
                                transformer_load_path=transformer_load_path,
                                dropout=0.1,
                                adv_alpha=adv_alpha,
                                r_drop_alpha=r_drop_alpha,
                                # action_weights=action_weights, 
                                # intent_weights=intent_weights
                                )
"""
Trainer
"""
def build_trainer(model: Model,
                  train_loader: DataLoader,
                  dev_loader: DataLoader,
                  serialization_dir: str,
                  cuda_device: torch.device,
                  num_epochs: int,
                  patience: int
                  ) -> Trainer:
    
    no_bigger = ["dialogue_encoder", "crf_act", "crf_int",
                 "act_decoder", "intent_decoder"]
    parameter_groups = [
    # 除了 no_bigger 中指定的模块以外的所有参数，不对这些参数应用权重衰减
    # weight_decay为0.0可以最小化正则化的影响,使BERT的权重更多地保持原有的值
    {
     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
     "weight_decay": 0.0,
    },
    # no_bigger中指定的模块的参数使用较大的学习率，在当前任务上从头开始训练的,使用较大的学习率可以加速其训练和收敛
    # 其他模块（BERT）在微调时使用较小的学习率可以防止过大的参数更新破坏已学到的知识
    {
     "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
     "lr": 0.0001
    }
    ]
    #
    optimizer = AdamW(parameter_groups, lr=1e-5, eps=1e-8)
    # 动态调整学习率: warmup_steps预热步数，在预热阶段，学习率会从 0 线性增加到初始学习率
    # 可以在训练初期让模型适应数据，避免过大的学习率导致不稳定
    lrschedule = LinearWithWarmup(optimizer=optimizer,
                                  num_epochs=num_epochs,
                                  num_steps_per_epoch=len(train_loader),
                                  warmup_steps=150)
    
    ckp = Checkpointer(serialization_dir=serialization_dir,
                       num_serialized_models_to_keep=-1)

    trainer = GradientDescentTrainer(model=model,
                                 optimizer=optimizer,
                                 data_loader=train_loader,
                                 patience=patience,
                                 validation_data_loader=dev_loader,
                                 validation_metric='+avg_accuracy',
                                 num_epochs=num_epochs,
                                 serialization_dir=serialization_dir,
                                 cuda_device=cuda_device if str(cuda_device) != 'cpu' else -1,
                                 learning_rate_scheduler=lrschedule,
                                 num_gradient_accumulation_steps=1,
                                 checkpointer=ckp,
                                 )

    
    return trainer

"""
数据增强：随机交换相邻句子的逻辑
"""
def swap_sentences(dialogue, speaker_ids, intentions, actions):
    # 创建副本
    swapped_dialogue = dialogue[:]
    swapped_speaker_ids = speaker_ids[:]
    swapped_intentions = intentions[:]
    swapped_actions = actions[:]

    if len(dialogue) > 1:
        # 交换1到min(3, len(dialogue)-1)之间
        num_swaps = random.randint(1, min(3, len(dialogue) - 1))
        for _ in range(num_swaps):
            idx1 = random.randint(0, len(dialogue) - 2)
            idx2 = idx1 + 1

            swapped_dialogue[idx1], swapped_dialogue[idx2] = swapped_dialogue[idx2], swapped_dialogue[idx1]
            swapped_speaker_ids[idx1], swapped_speaker_ids[idx2] = swapped_speaker_ids[idx2], swapped_speaker_ids[idx1]
            swapped_intentions[idx1], swapped_intentions[idx2] = swapped_intentions[idx2], swapped_intentions[idx1]
            swapped_actions[idx1], swapped_actions[idx2] = swapped_actions[idx2], swapped_actions[idx1]

    return swapped_dialogue, swapped_speaker_ids, swapped_intentions, swapped_actions
"""
随机插入无意义句子的逻辑
"""
def insert_meaningless_sentence(dialogue, speaker_ids, intentions, actions):
    inserted_dialogue = dialogue[:]
    inserted_speaker_ids = speaker_ids[:]
    inserted_intentions = intentions[:]
    inserted_actions = actions[:]

    num_inserts = random.randint(1, 3)
    for _ in range(num_inserts):
        insert_pos = random.randint(0, len(dialogue))

        meaningless_sentence = [random.randint(0, 10) for _ in range(random.randint(1, 10))]
        inserted_dialogue.insert(insert_pos, meaningless_sentence)

        previous_speaker = inserted_speaker_ids[insert_pos - 1] if insert_pos > 0 else inserted_speaker_ids[0]
        inserted_speaker_ids.insert(insert_pos, previous_speaker)

        previous_intention = inserted_intentions[insert_pos - 1] if insert_pos > 0 else inserted_intentions[0]
        inserted_intentions.insert(insert_pos, previous_intention)

        previous_action = inserted_actions[insert_pos - 1] if insert_pos > 0 else inserted_actions[0]
        inserted_actions.insert(insert_pos, previous_action)

    return inserted_dialogue, inserted_speaker_ids, inserted_intentions, inserted_actions
"""
随机mask
"""
def mask_and_predict(utterance, tokenizer, mask_prob=0.15):
    utterance_tokens = tokenizer.convert_ids_to_tokens(utterance)[1:-1]  # 去除 [CLS] 和 [SEP] token
    masked_utterance = ['[CLS]']

    for token in utterance_tokens:
        if random.random() < mask_prob:
            masked_utterance.append('[MASK]')
        else:
            masked_utterance.append(token)

    masked_utterance.append('[SEP]')
    masked_utterance = tokenizer.convert_tokens_to_ids(masked_utterance)  # 将字符串转换为数值索引
    return masked_utterance

def run_training_loop(config):
    logger = logging.getLogger(__name__)

    serialization_dir = config.output_model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    os.makedirs(serialization_dir, exist_ok=True)
    
    train_dataset_reader = IntentionRecognitionDatasetReader(transformer_load_path=config.pretrained_model_dir, 
                                                            training=False)  # 创建训练集的reader,启用数据增强
    dev_dataset_reader = IntentionRecognitionDatasetReader(transformer_load_path=config.pretrained_model_dir,
                                                        training=False)  # 创建验证集的reader,禁用数据增强

    train_path = config.train_file
    dev_path = config.dev_file

    train_data = list(train_dataset_reader.read(train_path))
    dev_data = list(dev_dataset_reader.read(dev_path))

    # # 对训练数据进行数据增强
    # augmented_train_data = []
    # for instance in train_data:
    #     augmented_train_data.append(instance)  # 添加原始数据

    #     dialogue = [utterance.array for utterance in instance.fields['dialogue'].field_list]
    #     speaker_ids = [speaker.label for speaker in instance.fields['speaker'].field_list]
    #     intentions = [intent.label for intent in instance.fields['intentions'].field_list]
    #     actions = [action.label for action in instance.fields['actions'].field_list]

    #     # 随机交换相邻句子
    #     if random.random() < 0.2:
    #         swapped_dialogue, swapped_speaker_ids, swapped_intentions, swapped_actions = swap_sentences(
    #             dialogue, speaker_ids, intentions, actions
    #         )
    #         augmented_train_data.append(Instance({
    #             'dialogue': ListField([TensorField(torch.tensor(u)) for u in swapped_dialogue]),
    #             'speaker': ListField([LabelField(s, label_namespace='speaker_labels') for s in swapped_speaker_ids]),
    #             'intentions': ListField([LabelField(i, label_namespace='intention_labels') for i in swapped_intentions]),
    #             'actions': ListField([LabelField(a, label_namespace='action_labels') for a in swapped_actions])
    #         }))

    #     # 随机插入无意义句子
    #     if random.random() < 0.2:
    #         inserted_dialogue, inserted_speaker_ids, inserted_intentions, inserted_actions = insert_meaningless_sentence(
    #             dialogue, speaker_ids, intentions, actions
    #         )
    #         augmented_train_data.append(Instance({
    #             'dialogue': ListField([TensorField(torch.tensor(u)) for u in inserted_dialogue]),
    #             'speaker': ListField([LabelField(s, label_namespace='speaker_labels') for s in inserted_speaker_ids]),
    #             'intentions': ListField([LabelField(i, label_namespace='intention_labels') for i in inserted_intentions]),
    #             'actions': ListField([LabelField(a, label_namespace='action_labels') for a in inserted_actions])
    #         }))

    #     # 随机mask
    #     masked_dialogue = [mask_and_predict(utterance, train_dataset_reader._transformer_indexers) for utterance in dialogue]
    #     augmented_train_data.append(Instance({
    #         'dialogue': ListField([TensorField(torch.tensor(u)) for u in masked_dialogue]),
    #         'speaker': instance.fields['speaker'],
    #         'intentions': instance.fields['intentions'],
    #         'actions': instance.fields['actions']
    #     }))

    # train_data = augmented_train_data


    all_data = train_data + dev_data
    vocab = build_vocab(all_data)
    vocab.save_to_files(vocabulary_dir)

    # 使用交叉验证
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
        logger.info(f"Fold {fold+1}/{k}")

        train_instances = [all_data[i] for i in train_idx]  
        val_instances = [all_data[i] for i in val_idx]

        # 对训练集数据进行数据增强
        augmented_train_instances = []
        for instance in train_instances:
            dialogue = [utterance.array for utterance in instance.fields['dialogue'].field_list]
            speaker_ids = [speaker.label for speaker in instance.fields['speaker'].field_list]
            intentions = [intent.label for intent in instance.fields['intentions'].field_list]
            actions = [action.label for action in instance.fields['actions'].field_list]

            # 随机交换相邻句子
            if random.random() < 0.5:
                m = 3
                idx1 = random.randint(0, len(dialogue)-1)
                idx2 = random.randint(max(0, idx1-m), min(len(dialogue)-1, idx1+m))

                swapped_dialogue = dialogue[:]
                swapped_speaker_ids = speaker_ids[:]
                swapped_intentions = intentions[:]
                swapped_actions = actions[:]

                swapped_dialogue[idx1], swapped_dialogue[idx2] = swapped_dialogue[idx2], swapped_dialogue[idx1]
                swapped_speaker_ids[idx1], swapped_speaker_ids[idx2] = swapped_speaker_ids[idx2], swapped_speaker_ids[idx1]
                swapped_intentions[idx1], swapped_intentions[idx2] = swapped_intentions[idx2], swapped_intentions[idx1]
                swapped_actions[idx1], swapped_actions[idx2] = swapped_actions[idx2], swapped_actions[idx1]

                augmented_train_instances.append(Instance({
                    'dialogue': ListField([TensorField(torch.tensor(u)) for u in swapped_dialogue]),
                    'speaker': ListField([LabelField(s, label_namespace='speaker_labels') for s in swapped_speaker_ids]),
                    'intentions': ListField([LabelField(i, label_namespace='intention_labels') for i in swapped_intentions]),
                    'actions': ListField([LabelField(a, label_namespace='action_labels') for a in swapped_actions])
                }))

            # 随机插入无意义句子
            if random.random() < 0.25:
                insert_pos = random.randint(0, len(dialogue))

                inserted_dialogue = dialogue[:]
                inserted_speaker_ids = speaker_ids[:]
                inserted_intentions = intentions[:]
                inserted_actions = actions[:]

                inserted_dialogue.insert(insert_pos, train_dataset_reader._transformer_indexers.convert_tokens_to_ids(['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[SEP]']))
                inserted_speaker_ids.insert(insert_pos, inserted_speaker_ids[insert_pos-1])
                inserted_intentions.insert(insert_pos, inserted_intentions[insert_pos-1])
                inserted_actions.insert(insert_pos, inserted_actions[insert_pos-1])

                augmented_train_instances.append(Instance({
                    'dialogue': ListField([TensorField(torch.tensor(u)) for u in inserted_dialogue]),
                    'speaker': ListField([LabelField(s, label_namespace='speaker_labels') for s in inserted_speaker_ids]),
                    'intentions': ListField([LabelField(i, label_namespace='intention_labels') for i in inserted_intentions]),
                    'actions': ListField([LabelField(a, label_namespace='action_labels') for a in inserted_actions])
                }))

            # 随机mask
            masked_dialogue = []
            for utterance in dialogue:
                utterance_tokens = train_dataset_reader._transformer_indexers.convert_ids_to_tokens(utterance)[1:-1]
                utterance_tokens = ['[CLS]'] + utterance_tokens + ['[SEP]']

                for j, token in enumerate(utterance_tokens):
                    if j == 0 or j == len(utterance_tokens) - 1:
                        continue
                    if random.random() < 0.25:
                        utterance_tokens[j] = '[MASK]'
                
                masked_dialogue.append(train_dataset_reader._transformer_indexers.convert_tokens_to_ids(utterance_tokens))

            augmented_train_instances.append(Instance({
                'dialogue': ListField([TensorField(torch.tensor(u)) for u in masked_dialogue]),
                'speaker': instance.fields['speaker'],
                'intentions': instance.fields['intentions'],
                'actions': instance.fields['actions']
            }))

        train_instances.extend(augmented_train_instances)  # 将增强后的数据添加到训练集中
        
        fold_serialization_dir = f"{serialization_dir}/fold{fold+1}"
        os.makedirs(fold_serialization_dir, exist_ok=True)

        vocab = build_vocab(train_instances+val_instances)
        vocab.save_to_files(fold_serialization_dir)

        train_loader = SimpleDataLoader(train_instances, batch_size=config.batch_size, shuffle=True)
        val_loader = SimpleDataLoader(val_instances, batch_size=config.batch_size, shuffle=False)

        
        train_loader.index_with(vocab)
        val_loader.index_with(vocab)

        device = torch.device(config.cuda_id if torch.cuda.is_available() else "cpu")
        model = build_model(vocab, config.pretrained_model_dir, config.pretrained_hidden_size,
                            config.adv_alpha, config.r_drop_alpha)
        model = model.to(device)



        trainer = build_trainer(model, train_loader, val_loader, fold_serialization_dir, 
                                device, config.num_epochs, config.patience)
        trainer.train()

    return trainer



    # 不使用交叉验证
    train_loader = SimpleDataLoader(augmented_train_data, batch_size=config.batch_size, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size=config.batch_size, shuffle=False) 
    # train_loader = MultiProcessDataLoader(train_dataset_reader, train_path, batch_size=config.batch_size, shuffle=True)
    # dev_loader = MultiProcessDataLoader(dev_dataset_reader, dev_path, batch_size=config.batch_size, shuffle=False)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    ## 解决不平衡标签问题
    # 统计训练数据集中每个类别的频率
    # action_counts = torch.zeros(vocab.get_vocab_size('action_labels'))
    # intent_counts = torch.zeros(vocab.get_vocab_size('intention_labels'))

    # for instance in train_data:
    #     actions = instance.fields['actions'].field_list
    #     for action in actions:
    #         action_counts[vocab.get_token_index(action.label, 'action_labels')] += 1
            
    #     intentions = instance.fields['intentions'].field_list
    #     for intention in intentions:
    #         intent_counts[vocab.get_token_index(intention.label, 'intention_labels')] += 1
            
    # # 计算权重        
    # action_weights = 1.0 / (action_counts + 1e-5)
    # intent_weights = 1.0 / (intent_counts + 1e-5)

    # # 归一化权重
    # action_weights = action_weights / action_weights.sum()
    # intent_weights = intent_weights / intent_weights.sum()
    
    device = torch.device(config.cuda_id if torch.cuda.is_available() else "cpu")
    # # 要放在GPU上,否则后面和log_likelihood_act计算会报不在同一个设备上
    # action_weights = action_weights.to(device)
    # intent_weights = intent_weights.to(device)
    # model = build_model(vocab, config.pretrained_model_dir, config.pretrained_hidden_size,
    #                 action_weights, intent_weights,
    #                 config.adv_alpha, config.r_drop_alpha)
    # model = build_model(vocab, config.pretrained_model_dir, config.pretrained_hidden_size,config.adv_alpha,config.r_drop_alpha)
    # model = model.to(device)
    
    # trainer = build_trainer(model,
    #                         train_loader,
    #                         dev_loader,
    #                         serialization_dir,
    #                         device,
    #                         config.num_epochs,
    #                         config.patience)
    # trainer.train()
    # return trainer

if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", default='./data/IMCS-DAC_train.json', type=str)
    parser.add_argument("--dev_file", default='./data/IMCS-DAC_dev.json', type=str)
    parser.add_argument("--output_model_dir", default='./save_model', type=str)
    parser.add_argument("--pretrained_model_dir", default='./plms/chinese-roberta-wwm-ext', type=str)
    parser.add_argument("--pretrained_hidden_size", default=768, type=int)
    parser.add_argument("--cuda_id", default='cuda:1', type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--patience", default=3, type=int)

    parser.add_argument("--adv_alpha", default=0.5, type=float, help="对抗学习loss权重") 
    parser.add_argument("--r_drop_alpha", default=0.5, type=float, help="R-Drop loss权重")
    
    config = parser.parse_args()
    run_training_loop(config)
    
    