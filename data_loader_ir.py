# -*- coding: utf-8 -*-
# data_loader.py

import json
import torch
import random
import logging
logger = logging.getLogger(__name__)

from typing import Dict, List
from overrides import overrides
from transformers import BertTokenizer

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TensorField, LabelField, ListField

SPECIAL_TOKENS = {'患者':'[unused1]', '医生':'[unused2]'}
SPECIAL_LABELS = {'Other', 'Diagnose'}

# WINDOW = 30 # Decided by my GPU Restriction，动态窗口的逻辑

class IntentionRecognitionDatasetReader(DatasetReader):
    def __init__(self, transformer_load_path: str, training: bool = False, **kwargs, # 默认不开启数据增强
    ) -> None:
        super().__init__(**kwargs)
        self._transformer_indexers = BertTokenizer.from_pretrained(transformer_load_path)
        self.training = training
    
    @overrides
    def _read(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            data_file = json.load(file)
            for eid in data_file.keys():
                dialogue, speaker_ids, intentions, actions = [], [], [], []
                # for sid in data_file[eid]['dialogue']:
                for sid in data_file[eid]:
                    speaker_ids.append(sid['speaker'])
                    speaker = [SPECIAL_TOKENS[sid['speaker']]]
                    utterance = list(sid['sentence'])
                    utterance = speaker + utterance
                    dialogue.append(utterance)
                    if sid['dialogue_act'] not in SPECIAL_LABELS:
                        intention, action = sid['dialogue_act'].split('-')
                    else:
                        intention = sid['dialogue_act']
                        action = sid['dialogue_act']
                    intentions.append(intention)
                    actions.append(action)
                # 动态 Batch
                yield self.text_to_instance(dialogue, speaker_ids, intentions, actions)
                
                # If you have sufficient GPU Memory, Put Whole Dialogue in will be better.
                # for i in range(0, len(dialogue), WINDOW):
                #     y = i + WINDOW
                #     yield self.text_to_instance(dialogue[i:y],
                #                                 speaker_ids[i:y],
                #                                 intentions[i:y],
                #                                 actions[i:y])
                #     if y >= len(dialogue):
                #         break

# 有点问题    
    def text_to_instance(
        self,
        dialogue: List[List[str]],
        speaker_ids: List[str],
        intentions: List[str] = None,
        actions: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        if self.training: 
            # 随机交换相邻句子
            if random.random() < 0.2: 
                dialogue, speaker_ids, intentions, actions = self.swap_sentences(dialogue, speaker_ids, intentions, actions)
            
            # 随机插入无意义句子
            if random.random() < 0.2:
                insert_pos = random.randint(0, len(dialogue))
                dialogue.insert(insert_pos, ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[SEP]'])
                speaker_ids.insert(insert_pos, speaker_ids[insert_pos-1]) 
                intentions.insert(insert_pos, intentions[insert_pos-1])
                actions.insert(insert_pos, actions[insert_pos-1])

            # 随机mask
            dialogue = [self.mask_and_predict(utterance) for utterance in dialogue]
        else:
            dialogue = [['[CLS]'] + utterance + ['[SEP]'] for utterance in dialogue]

        dialogue_field = [self._transformer_indexers.convert_tokens_to_ids(utterance) for utterance in dialogue]
        dialogue_field = [TensorField(torch.tensor(u)) for u in dialogue_field]
        fields["dialogue"] = ListField(dialogue_field)
        speaker_field = [LabelField(speaker, label_namespace='speaker_labels') for speaker in speaker_ids]
        fields["speaker"] = ListField(speaker_field)
        if intentions != None:
            intents_field = [LabelField(intention, label_namespace='intention_labels') for intention in intentions]
            fields["intentions"] = ListField(intents_field)
        if actions != None:
            actions_field = [LabelField(action, label_namespace='action_labels') for action in actions]
            fields["actions"] = ListField(actions_field)

        return Instance(fields)
    
    def swap_sentences(self, dialogue, speaker_ids, intentions, actions):
        # 随机选择一个句子,与前后m个句子交换
        m = 3 
        idx1 = random.randint(0, len(dialogue)-1)
        idx2 = random.randint(max(0, idx1-m), min(len(dialogue)-1, idx1+m))
        
        dialogue[idx1], dialogue[idx2] = dialogue[idx2], dialogue[idx1] 
        speaker_ids[idx1], speaker_ids[idx2] = speaker_ids[idx2], speaker_ids[idx1]
        if intentions is not None:
            intentions[idx1], intentions[idx2] = intentions[idx2], intentions[idx1]
        if actions is not None:
            actions[idx1], actions[idx2] = actions[idx2], actions[idx1]
            
        return dialogue, speaker_ids, intentions, actions


    def mask_and_predict(self, utterance):
        # 随机mask词语
        utterance = ['[CLS]'] + utterance + ['[SEP]'] 
        for i, token in enumerate(utterance):
            if i == 0 or i == len(utterance) - 1:
                continue
            if random.random() < 0.15: 
                utterance[i] = '[MASK]'
        return utterance

