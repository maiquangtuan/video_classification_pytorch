import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T

torch.manual_seed(97)
num_frames = 16 # 16
clip_steps = 2
Bs_Train = 16
Bs_Test = 16

transform = transforms.Compose([  
                                                     
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((200, 200)),
                                 T.RandomHorizontalFlip(),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((172, 172))])
transform_test = transforms.Compose([                           
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((200, 200)),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.CenterCrop((172, 172))])

hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,frame_rate=5,
                                                step_between_clips = clip_steps, fold=1, train=True,
                                                transform=transform, num_workers=2)


hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,frame_rate=5,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=transform_test, num_workers=2)
train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
test_loader  = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)
