import torch
from torch.utils.data import Dataset, DataLoader
from .baseline_data import FakeSVDataset, FakeTTDataset, FVCDataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

class FakeSVDataset_ExMRD(FakeSVDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FakeSVDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.lm_ocr = pd.read_json(f'data/FakeSV/CoT/{lm}/lm_text_refine.jsonl', lines=True)
        self.caption = pd.read_json(f'data/FakeSV/CoT/{lm}/lm_visual_refine.jsonl', lines=True)
        self.lm_comsense = pd.read_json(f'data/FakeSV/CoT/{lm}/lm_retrieve.jsonl', lines=True)
        self.lm_causal = pd.read_json(f'data/FakeSV/CoT/{lm}/lm_reason.jsonl', lines=True)
    
        self.fea_frames = torch.load('data/FakeSV/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        # response

        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)
        
        text_lm_ocr = f'{item["title"]} {text_lm_ocr}'

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)

        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FakeSVCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])
        
        max_len = 256
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'comsense_input': comsense_input,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'causal_input': causal_input,
        }

class FakeTTDataset_ExMRD(FakeTTDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FakeTTDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.ocr = pd.read_json('data/FakeTT/ocr.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_ocr = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
     
        self.lm_comsense = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FakeTT/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['description']}\n{text_lm_ocr}'
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)

        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FakeTTCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = 77
        # texts_input = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }


class FVCDataset_ExMRD(FVCDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FVCDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.lm_ocr = pd.read_json(f'data/FVC/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FVC/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
         
        self.lm_comsense = pd.read_json(f'data/FVC/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FVC/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FVC/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['title']}\n{item['description']}\n{text_lm_ocr}'
        
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)
        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FVCCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = 77
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }