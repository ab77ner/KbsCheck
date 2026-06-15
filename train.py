import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import AuditDataset, compute_class_weights
from src.metrics import compute_metrics, format_multiclass_report
from src.model import CNNBiLSTMAttention
from src.plotting import plot_confusion_matrix, plot_roc_curves
from src.utils import set_seed, get_device, load_config, ensure_dir, save_json


class EarlyStopping:
    def __init__(self, patience=300, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.counter=0
    def step(self, value):
        if value < self.best - self.min_delta:
            self.best=value; self.counter=0; return False
        self.counter += 1
        return self.counter >= self.patience


def run_eval(model, loader, criterion, device, class_names):
    model.eval(); losses=[]; y_true=[]; y_pred=[]; y_prob=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device).float(); y=y.to(device).long()
            logits=model(x); loss=criterion(logits,y); probs=torch.softmax(logits,dim=1)
            losses.append(float(loss.item()))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(probs,dim=1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    m=compute_metrics(y_true,y_pred,np.asarray(y_prob),class_names)
    m['loss']=float(np.mean(losses)) if losses else 0.0
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_config.yaml')
    ap.add_argument('--train_path', default=None)
    ap.add_argument('--val_path', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--device', default=None)
    args=ap.parse_args()
    cfg=load_config(args.config); set_seed(cfg.get('seed',42))
    data_cfg=cfg.get('data',{}); model_cfg=cfg.get('model',{}); train_cfg=cfg.get('train',{}); out_cfg=cfg.get('outputs',{})
    train_path=args.train_path or data_cfg.get('train_path','data/train.jsonl')
    val_path=args.val_path or data_cfg.get('val_path','data/val.jsonl')
    epochs=args.epochs or train_cfg.get('epochs',1000)
    batch_size=args.batch_size or train_cfg.get('batch_size',96)
    lr=args.learning_rate or train_cfg.get('learning_rate',1e-3)
    device=get_device(args.device or train_cfg.get('device','auto'))
    class_names=cfg.get('class_names',[str(i) for i in range(cfg.get('num_classes',8))])
    num_classes=len(class_names); seq_len=data_cfg.get('seq_len',20); input_dim=data_cfg.get('input_dim',19)
    output_dir=out_cfg.get('output_dir','outputs'); plot_dir=out_cfg.get('plot_dir','plots')
    ensure_dir(output_dir); ensure_dir(plot_dir)
    train_ds=AuditDataset(train_path,seq_len=seq_len,input_dim=input_dim)
    val_ds=AuditDataset(val_path,seq_len=seq_len,input_dim=input_dim)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model=CNNBiLSTMAttention(input_dim=input_dim,embed_dim=model_cfg.get('embed_dim',96),conv_channels=model_cfg.get('conv_channels',128),hidden_dim=model_cfg.get('hidden_dim',160),num_classes=num_classes,dropout=model_cfg.get('dropout',0.25)).to(device)
    if train_cfg.get('use_class_weight',True): criterion=nn.CrossEntropyLoss(weight=compute_class_weights(train_ds,num_classes).to(device))
    else: criterion=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=train_cfg.get('weight_decay',1e-5))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=30,factor=0.5)
    early=EarlyStopping(train_cfg.get('patience',300),train_cfg.get('min_delta',1e-4))
    best=float('inf'); history=[]; grad_clip=train_cfg.get('grad_clip',5.0)
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for x,y in train_loader:
            try:
                x=x.to(device).float(); y=y.to(device).long()
                if int(y.max())>=num_classes or int(y.min())<0: continue
                logits=model(x); loss=criterion(logits,y)
                if torch.isnan(loss) or torch.isinf(loss): continue
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); opt.step(); losses.append(float(loss.item()))
            except Exception as e:
                print(f'skip bad batch: {e}')
        train_loss=float(np.mean(losses)) if losses else 0.0
        val_metrics=run_eval(model,val_loader,criterion,device,class_names); scheduler.step(val_metrics['loss'])
        row={'epoch':epoch,'train_loss':train_loss,'val_loss':val_metrics['loss'],'val_accuracy':val_metrics['accuracy'],'val_precision':val_metrics['precision'],'val_precision_macro':val_metrics['precision_macro'],'val_recall':val_metrics['recall'],'val_recall_macro':val_metrics['recall_macro'],'val_f1':val_metrics['f1'],'val_f1_macro':val_metrics['f1_macro'],'val_auc':val_metrics['auc']}
        history.append(row); print(json.dumps(row,ensure_ascii=False))
        if val_metrics['loss'] < best:
            best=val_metrics['loss']
            torch.save({'model_state':model.state_dict(),'config':cfg,'class_names':class_names,'input_dim':input_dim,'seq_len':seq_len}, Path(output_dir)/'best_model.pt')
            save_json(val_metrics, Path(output_dir)/'best_val_metrics.json')
            plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, Path(plot_dir)/'val_confusion_matrix.png')
            plot_roc_curves(val_metrics['roc'], Path(plot_dir)/'val_roc_curve.png')
        if early.step(val_metrics['loss']):
            print(f'Early stopping triggered at epoch {epoch}')
            break
    save_json(history, Path(output_dir)/'training_history.json')
    print(f"best model saved to {Path(output_dir)/'best_model.pt'}")

if __name__ == '__main__':
    main()
