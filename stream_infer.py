import argparse
import json
import time
from collections import deque
import numpy as np
import torch
from src.features import parse_audit_event, event_to_feature
from src.model import CNNBiLSTMAttention
from src.utils import get_device


def follow(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        f.seek(0,2)
        while True:
            line=f.readline()
            if not line:
                time.sleep(0.2); continue
            yield line


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',default='outputs/best_model.pt'); ap.add_argument('--audit_log',required=True); ap.add_argument('--device',default='auto'); args=ap.parse_args()
    device=get_device(args.device); ckpt=torch.load(args.ckpt,map_location=device); cfg=ckpt.get('config',{})
    class_names=ckpt.get('class_names',cfg.get('class_names',[str(i) for i in range(8)])); seq_len=ckpt.get('seq_len',20); input_dim=ckpt.get('input_dim',19); model_cfg=cfg.get('model',{})
    model=CNNBiLSTMAttention(input_dim=input_dim,embed_dim=model_cfg.get('embed_dim',96),conv_channels=model_cfg.get('conv_channels',128),hidden_dim=model_cfg.get('hidden_dim',160),num_classes=len(class_names),dropout=model_cfg.get('dropout',0.25)).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval(); buf=deque(maxlen=seq_len); prev=None
    for line in follow(args.audit_log):
        ev=parse_audit_event(line)
        if ev is None: continue
        feat=event_to_feature(ev,prev); prev=ev; buf.append(feat)
        if len(buf)<seq_len: continue
        x=torch.tensor(np.asarray(list(buf),dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad(): prob=torch.softmax(model(x),dim=1).squeeze().cpu().numpy()
        pred=int(np.argmax(prob))
        print(json.dumps({'pred_label':pred,'pred_name':str(class_names[pred]),'confidence':float(prob[pred]),'user':ev.get('user'),'verb':ev.get('verb'),'resource':ev.get('resource'),'status_code':ev.get('status_code')},ensure_ascii=False))

if __name__ == '__main__':
    main()
