import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
from vivit import ViViT
from torch import nn
from train import train_iter, evaluate_test, evaluate_val
import argparse
from data import *
from pytorch_pretrained_vit import ViT
from models import *
import wandb
wandb.login()

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train vivit model')

parser.add_argument('--num_frames', type=int, default = 16)
parser.add_argument('--batch_size', type=int, default = 16)
parser.add_argument('--ratio', type=float, default=0.85)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--num_epochs', type=int, default = 50)
parser.add_argument('--dropout', type=float, default = 0.5)
parser.add_argument('--model_name', type=str, choices=['vivit', 'movinet'])
parser.add_argument('--vit_model_name', type = str, default = 'B_16', choices = ['B_16', 'B_32', 'L_16', 'L_32' ])

parser.add_argument('--save_dir', type = str, default = r'./save_dir')

def main():
    # parsing args
    
    args = parser.parse_args()
    
    if args.model_name == 'vivit':
        base_model = ViT(args.vivit_model_name, pretrained=True)
        model = ViViT(image_size=224, patch_size=16, num_classes=51, num_frames=args.num_frames, pretrain_model=base_model)
        model = nn.DataParallel(model)
    elif args.model_name == 'movinet':
        if args.model_name == 'A0':
            model = MoViNet(_C.MODEL.MoViNetA0, 600,causal = False, pretrained = True, tf_like = True )
        elif args.model_name == 'A1':
            model = MoViNet(_C.MODEL.MoViNetA1, 600,causal = False, pretrained = True, tf_like = True )
            model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))
            model = nn.DataParallel(model)

    
    trloss_val,vlloss_val, tsloss_val =[], [], []   #list of loss function value

    optimz = optim.Adam(model.parameters(), lr=args.lr)
    run_id = f'{args.data}_B{args.batch_size}_T{args.num_frames}'
    wandb.init(project="vivit-aggression", mode = 'online', id = run_id)

    train_iter(args, model, optimz, dataloaders['train'],dataloaders['valid'],dataloaders['test'],trloss_val,vlloss_val, tsloss_val )
        
        
    save_path = f'{args.save_dir}/vivit_{args.num_epochs}.pt'
    torch.save(model.state_dict(), save_path)
            
 
    
if __name__ == '__main__':
    main()