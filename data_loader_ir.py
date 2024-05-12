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

SPECIAL_TOKENS = {'患者': '[unused1]', '医生': '[unused2]'}  # 标识数据集中的角色信息
SPECIAL_LABELS = {'Other', 'Diagnose'}  # 标识数据集中不需要拆分的标签(特殊标签)


# WINDOW = 30 # 如果GPU显存不足，则使用滑动窗口的逻辑

class IntentionRecognitionDatasetReader(DatasetReader):
    def __init__(self, transformer_load_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._transformer_indexers = BertTokenizer.from_pretrained(transformer_load_path)

    @overrides
    def _read(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            data_file = json.load(file)
            #  遍历data_file中的每个对话,eid为对话的ID
            for eid in data_file.keys():
                dialogue, speaker_ids, intentions, actions = [], [], [], []
                #  遍历当前对话中的每个utterance,sid为utterance的ID
                for sid in data_file[eid]:
                    speaker_ids.append(sid['speaker'])  # 说话人ID
                    speaker = [SPECIAL_TOKENS[sid['speaker']]]  # 根据说话人ID,获取对应的特殊token
                    utterance = list(sid['sentence'])
                    utterance = speaker + utterance  # 将speaker和utterance拼接在一起,组成完整的utterance
                    dialogue.append(utterance)
                    #  判断是否为特殊标签
                    if sid['dialogue_act'] not in SPECIAL_LABELS:
                        intention, action = sid['dialogue_act'].split('-')
                    else:
                        intention = sid['dialogue_act']
                        action = sid['dialogue_act']
                    intentions.append(intention)
                    actions.append(action)
                # 1.显存充足：直接返回整个对话示例给DatasetReader，实现动态 Batch
                yield self.text_to_instance(dialogue, speaker_ids, intentions, actions)

                # 2.显存不充足：以窗口的方式滑动着返回部分对话，但对性能有影响
                # for i in range(0, len(dialogue), WINDOW):
                #     y = i + WINDOW
                #     yield self.text_to_instance(dialogue[i:y],
                #                                 speaker_ids[i:y],
                #                                 intentions[i:y],
                #                                 actions[i:y])
                #     if y >= len(dialogue):
                #         break

    """
    将对话转换为Instance对象
    """
    def text_to_instance(
            self,
            dialogue: List[List[str]],
            speaker_ids: List[str],
            intentions: List[str] = None,
            actions: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        # 为每个utterance添加BERT的特殊token[CLS]和[SEP]
        dialogue = [['[CLS]'] + utterance + ['[SEP]'] for utterance in dialogue]
        # 使用BERT tokenizer将每个utterance转换为对应的token ID列表
        dialogue_field = [self._transformer_indexers.convert_tokens_to_ids(utterance) for utterance in dialogue]
        # 将每个utterance的token ID列表转换为TensorField对象
        dialogue_field = [TensorField(torch.tensor(u)) for u in dialogue_field]
        fields["dialogue"] = ListField(dialogue_field)
        # 将每个utterance的说话人ID转换为LabelField对象
        speaker_field = [LabelField(speaker, label_namespace='speaker_labels') for speaker in speaker_ids]
        fields["speaker"] = ListField(speaker_field)
        # 标签不为空
        if intentions is not None:
            intents_field = [LabelField(intention, label_namespace='intention_labels') for intention in intentions]
            fields["intentions"] = ListField(intents_field)
        if actions is not None:
            actions_field = [LabelField(action, label_namespace='action_labels') for action in actions]
            fields["actions"] = ListField(actions_field)

        return Instance(fields)
