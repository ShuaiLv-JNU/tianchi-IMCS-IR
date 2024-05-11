# -*- coding: utf-8 -*-
# predict.py

import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from trainer import build_model
from transformers import BertTokenizer
from data_loader_ir import IntentionRecognitionDatasetReader, SPECIAL_TOKENS, SPECIAL_LABELS

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

logger = logging.getLogger(__name__)

class IRPredictor(Predictor):    
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 transformer_load_path : str
                 ) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab
        self._transformer_indexers = BertTokenizer.from_pretrained(transformer_load_path)
        
    def predict(self, dialogue, speaker_ids) -> JsonDict:
        result = self.predict_json({"dialogue": dialogue, "speaker": speaker_ids})
        instances = dict()
        instances['actions'] = [self.vocab.get_token_from_index(i,namespace='action_labels') for i in result['actions']]
        instances['intentions'] = [self.vocab.get_token_from_index(i,namespace='intention_labels') for i in result['intentions']]
        return instances

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        dialogue = json_dict["dialogue"]
        speaker_ids = json_dict["speaker"]
        return self._dataset_reader.text_to_instance(dialogue, speaker_ids)

def read_input_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    eids, dialogues, speaker_ids, sentences, sentence_ids = [], [], [], [], []
    for eid, dialogue in data.items():
        dialogue_, speaker_id, sentence, sentence_id = [], [], [], []
        for utt in dialogue:
            speaker = [SPECIAL_TOKENS[utt['speaker']]]
            utterance = speaker + list(utt['sentence'])
            dialogue_.append(utterance)
            speaker_id.append(utt['speaker'])
            sentence.append(utt['sentence'])
            sentence_id.append(utt['sentence_id'])
        eids.append(eid)
        dialogues.append(dialogue_)
        speaker_ids.append(speaker_id)
        sentences.append(sentence)
        sentence_ids.append(sentence_id)
    return eids, dialogues, speaker_ids, sentences, sentence_ids
def predict(pred_config):
    # 加载每个fold的最优模型
    k = 5
    predictors = []
    for fold in range(k):
        fold_dir = os.path.join(pred_config.model_dir, f"fold{fold+1}")
        vocab_dir = os.path.join(fold_dir, "vocabulary")
        vocab = Vocabulary.from_files(vocab_dir)
        model_dir = os.path.join(fold_dir, pred_config.model_name)
        model = build_model(vocab, pred_config.pretrained_model_dir,
                            pred_config.pretrained_hidden_size, adv_alpha=0.0, r_drop_alpha=0.0)
        device = torch.device(pred_config.cuda_id if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_dir, map_location=device))
        model = model.to(device)
        dataset_reader = IntentionRecognitionDatasetReader(transformer_load_path=pred_config.pretrained_model_dir,
                                                           training=False)
        predictor = IRPredictor(model=model,
                                dataset_reader=dataset_reader,
                                transformer_load_path=pred_config.pretrained_model_dir)
        predictors.append(predictor)

    # 加载原始的JSON数据文件
    with open(pred_config.test_input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    eids, dialogues, speaker_ids, sentences, sentence_ids = read_input_file(pred_config.test_input_file)
    predict_result = {eid: dialogue for eid, dialogue in data.items()}
    predict_subres = {eid: {} for eid in eids}

    for i in tqdm(range(len(eids))):
        fold_results = []
        for predictor in predictors:
            result = predictor.predict(dialogues[i], speaker_ids[i])
            fold_results.append(result)

        # 对所有fold的预测结果进行平均
        avg_result = {}
        for key in fold_results[0].keys():
            avg_result[key] = [max(pred[key][j] for pred in fold_results) for j in range(len(fold_results[0][key]))]

        acts, ints = avg_result['actions'], avg_result['intentions']

        for j, (utt, act, intent) in enumerate(zip(data[eids[i]], acts, ints)):
            idx = str(j+2) if eids[i]=='10708561' and j+1>=43 else str(j+1)
            if int(utt['sentence_id']) != int(idx):
                print(f"Sent id mismatch: {int(utt['sentence_id'])} vs {int(idx)}")
            utt['dialogue_act'] = f"{intent}-{act}" if act not in SPECIAL_LABELS and intent not in SPECIAL_LABELS else act if act in SPECIAL_LABELS else intent
            predict_subres[eids[i]][idx] = {'act': act, 'int': intent}

    pred_path = os.path.join(pred_config.test_output_file)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(predict_result, json_file, ensure_ascii=False, indent=4)
    pred_path_sub = os.path.join(pred_config.test_output_file + '.sub')
    with open(pred_path_sub, 'w', encoding='utf-8') as json_file:
        json.dump(predict_subres, json_file, ensure_ascii=False, indent=4)
    logger.info("Prediction Done!")
# def predict(pred_config):
#     serialization_dir = pred_config.model_dir
#     vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
#     vocab = Vocabulary.from_files(vocabulary_dir)
    
#     model_dir = os.path.join(serialization_dir, pred_config.model_name)
#     model = build_model(vocab, pred_config.pretrained_model_dir, pred_config.pretrained_hidden_size,adv_alpha=0.0,r_drop_alpha=0.0)
#     device = torch.device(pred_config.cuda_id if torch.cuda.is_available() else "cpu")
#     model.load_state_dict(torch.load(model_dir, map_location=device))
#     model = model.to(device)
    
#     dataset_reader = IntentionRecognitionDatasetReader(transformer_load_path=pred_config.pretrained_model_dir, training=False)# 不开启数据增强
#     predictor = IRPredictor(model=model,
#                             dataset_reader=dataset_reader,
#                             transformer_load_path=pred_config.pretrained_model_dir)
#     # 加载原始的JSON数据文件
#     with open(pred_config.test_input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     eids, dialogues, speaker_ids, sentences, sentence_ids = read_input_file(pred_config.test_input_file)
#     predict_result = {eid: dialogue for eid, dialogue in data.items()}
#     predict_subres = {eid: {} for eid in eids}  # 定义predict_subres字典
#     for i in tqdm(range(len(eids))):
#         result = predictor.predict(dialogues[i], speaker_ids[i])
#         acts, ints = result['actions'], result['intentions']
#         for j, (utt, act, intent) in enumerate(zip(data[eids[i]], acts, ints)):
#             idx = str(j+2) if eids[i]=='10708561' and j+1>=43 else str(j+1)
#             if int(utt['sentence_id']) != int(idx):
#                 print(f"Sent id mismatch: {int(utt['sentence_id'])} vs {int(idx)}")
#             utt['dialogue_act'] = f"{intent}-{act}" if act not in SPECIAL_LABELS and intent not in SPECIAL_LABELS else act if act in SPECIAL_LABELS else intent
#             predict_subres[eids[i]][idx] = {'act': act, 'int': intent}  # 将act和intent存入predict_subres
    
#     pred_path = os.path.join(pred_config.test_output_file)
#     with open(pred_path, 'w', encoding='utf-8') as json_file:
#         json.dump(predict_result, json_file, ensure_ascii=False, indent=4)
#     pred_path_sub = os.path.join(pred_config.test_output_file + '.sub')
#     with open(pred_path_sub, 'w', encoding='utf-8') as json_file:
#         json.dump(predict_subres, json_file, ensure_ascii=False, indent=4)
#     logger.info("Prediction Done!")

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_input_file", default="./data/IMCS-DAC_test.json", type=str)
    parser.add_argument("--test_output_file", default="IMCS-IR_test.json", type=str)
    parser.add_argument("--model_dir", default="./save_model", type=str)
    parser.add_argument("--model_name", default="best.th", type=str)
    parser.add_argument("--pretrained_model_dir", default="./plms/chinese-roberta-wwm-ext", type=str)
    parser.add_argument("--pretrained_hidden_size", default=768, type=int)
    parser.add_argument("--cuda_id", default='cuda:1', type=str)
    
    pred_config = parser.parse_args()
    predict(pred_config)
    
    