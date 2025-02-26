import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel
import torch
from loguru import logger
import math


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to human-readable format
    def human_readable(num):
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000:
                return f"{num:.1f}{unit}"
            num /= 1000
    
    print(f"Total parameters: {human_readable(total_params)}")
    print(f"Trainable parameters: {human_readable(trainable_params)}")


class BERT_FT(nn.Module):
    def __init__(self, text_encoder, num_frozen_layers):
        super(BERT_FT, self).__init__()
        self.bert = AutoModel.from_pretrained(text_encoder).text_model

        for name, param in self.bert.named_parameters():
            # print(name)
            if 'embeddings' in name or 'final_layer_norm' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                # get the layer index
                layer_idx = int(name.split('.')[2])
                if layer_idx >= num_frozen_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, **inputs):
        return self.bert(**inputs)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        
        self.init_weights()
    def init_weights(self):
        position = torch.arange(0, self.positional_encoding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.positional_encoding.size(1), 2) * 
                             -(math.log(10000.0) / self.positional_encoding.size(1)))
        
        pe = torch.zeros_like(self.positional_encoding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding.data.copy_(pe)
    def forward(self, x):
        return x + self.positional_encoding[:x.size(1), :]

class ExMRD(nn.Module):
    def __init__(self, hid_dim, dropout, text_encoder, num_frozen_layers=12, ablation='No'):
        super(ExMRD, self).__init__()
        self.name = 'ExMRD'
        self.text_encoder = text_encoder
        if ablation == 'w/o-finetune':
            num_frozen_layers = 12
        self.bert = BERT_FT(text_encoder, num_frozen_layers)
        print_model_params(self.bert)

        self.linear_ocr = nn.Sequential(nn.LazyLinear(hid_dim))
        self.linear_caption = nn.Sequential(nn.LazyLinear(hid_dim))

        self.linear_video = nn.Sequential(nn.LazyLinear(hid_dim))
        self.linear_comsense = nn.Sequential(nn.LazyLinear(hid_dim))
        self.linear_causal = nn.Sequential(nn.LazyLinear(hid_dim))


        self.classifier = nn.LazyLinear(2)
        self.temporal_pe = LearnablePositionalEncoding(1024, 16)

        self.ablation = ablation
        
    def forward(self, **inputs):
        self.bert.eval()
        fea_frames = inputs['fea_frames']

        lm_ocr_input = inputs['lm_ocr_input']
        caption_input = inputs['caption_input']
        comsense_input = inputs['comsense_input']
        causal_input = inputs['causal_input']
        
        if 'chinese' in self.text_encoder:
            fea_ocr = self.bert(**lm_ocr_input)['last_hidden_state'][:,0,:]
            fea_caption = self.bert(**caption_input)['last_hidden_state'][:,0,:]
            fea_comsense = self.bert(**comsense_input)['last_hidden_state'][:,0,:]
            fea_causal = self.bert(**causal_input)['last_hidden_state'][:,0,:]
        else:
            fea_ocr = self.bert(**lm_ocr_input)['pooler_output']
            fea_caption = self.bert(**caption_input)['pooler_output']
            fea_comsense = self.bert(**comsense_input)['pooler_output']
            fea_causal = self.bert(**causal_input)['pooler_output']
        
        fea_ocr = self.linear_ocr(fea_ocr)
        fea_caption = self.linear_caption(fea_caption)
        fea_comsense = self.linear_comsense(fea_comsense)
        fea_causal = self.linear_causal(fea_causal)

        fea_x = torch.concat((fea_ocr.unsqueeze(1), fea_caption.unsqueeze(1), fea_comsense.unsqueeze(1), fea_causal.unsqueeze(1)), 1)

        # add temporal pe to vitfeature
        fea_frames = self.temporal_pe(fea_frames)
        fea_frames = self.linear_video(fea_frames)
        
        fea_x = torch.mean(fea_x, dim=1)
        fea_frames = torch.mean(fea_frames, dim=1)
        fea = torch.stack((fea_frames, fea_x), 1)
        fea = torch.mean(fea, dim=1)
        
        output = self.classifier(fea)
        return output