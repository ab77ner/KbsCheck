import argparse
import json
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from src.features import parse_audit_event, event_to_feature, rule_label_event
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
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',default='outputs/best_model.pt'); ap.add_argument('--audit_log',required=True); ap.add_argument('--batch_sequences',type=int,default=16); ap.add_argument('--learning_rate',type=float,default=1e-5); ap.add_argument('--save_every',type=int,default=20); ap.add_argument('--device',default='auto'); args=ap.parse_args()
    device=get_device(args.device); ckpt=torch.load(args.ckpt,map_location=device); cfg=ckpt.get('config',{})
    class_names=ckpt.get('class_names',cfg.get('class_names',[str(i) for i in range(8)])); seq_len=ckpt.get('seq_len',20); input_dim=ckpt.get('input_dim',19); model_cfg=cfg.get('model',{})
    model=CNNBiLSTMAttention(input_dim=input_dim,embed_dim=model_cfg.get('embed_dim',96),conv_channels=model_cfg.get('conv_channels',128),hidden_dim=model_cfg.get('hidden_dim',160),num_classes=len(class_names),dropout=model_cfg.get('dropout',0.25)).to(device)
    model.load_state_dict(ckpt['model_state']); model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate); criterion=nn.CrossEntropyLoss()
    event_buf=deque(maxlen=seq_len); feat_buf=deque(maxlen=seq_len); train_x=[]; train_y=[]; prev=None; updates=0
    for line in follow(args.audit_log):
        ev=parse_audit_event(line)
        if ev is None: continue
        feat=event_to_feature(ev,prev); prev=ev; event_buf.append(ev); feat_buf.append(feat)
        if len(feat_buf)<seq_len: continue
        labels=[rule_label_event(e) for e in event_buf]; pseudo=max(labels)
        train_x.append(list(feat_buf)); train_y.append(pseudo)
        if len(train_x)>=args.batch_sequences:
            x=torch.tensor(np.asarray(train_x,dtype=np.float32)).to(device); y=torch.tensor(train_y,dtype=torch.long).to(device)
            logits=model(x); loss=criterion(logits,y); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); updates+=1
            print(json.dumps({'update':updates,'loss':float(loss.item())},ensure_ascii=False)); train_x=[]; train_y=[]
            if updates % args.save_every == 0:
                ckpt['model_state']=model.state_dict(); torch.save(ckpt,'outputs/online_finetuned_model.pt'); print('saved outputs/online_finetuned_model.pt')

if __name__ == '__main__':
    main()
