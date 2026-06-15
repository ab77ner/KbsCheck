import argparse
import json
import numpy as np
import torch
from src.dataset import AuditDataset
from src.model import CNNBiLSTMAttention
from src.utils import get_device


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/best_model.pt')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='outputs/predictions.jsonl')
    ap.add_argument('--device', default='auto')
    args=ap.parse_args()
    device=get_device(args.device); ckpt=torch.load(args.ckpt,map_location=device); cfg=ckpt.get('config',{})
    class_names=ckpt.get('class_names',cfg.get('class_names',[str(i) for i in range(8)])); input_dim=ckpt.get('input_dim',19); seq_len=ckpt.get('seq_len',20); model_cfg=cfg.get('model',{})
    model=CNNBiLSTMAttention(input_dim=input_dim,embed_dim=model_cfg.get('embed_dim',96),conv_channels=model_cfg.get('conv_channels',128),hidden_dim=model_cfg.get('hidden_dim',160),num_classes=len(class_names),dropout=model_cfg.get('dropout',0.25)).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval(); ds=AuditDataset(args.data,seq_len=seq_len,input_dim=input_dim)
    with open(args.out,'w',encoding='utf-8') as f:
        with torch.no_grad():
            for i in range(len(ds)):
                x,y=ds[i]; prob=torch.softmax(model(x.unsqueeze(0).to(device).float()),dim=1).squeeze().cpu().numpy(); pred=int(np.argmax(prob))
                f.write(json.dumps({'index':i,'true_label':int(y.item()),'pred_label':pred,'pred_name':str(class_names[pred]),'prob':prob.tolist()},ensure_ascii=False)+'\n')
    print(f'saved predictions to {args.out}')

if __name__ == '__main__':
    main()
