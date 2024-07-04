# Unet is our proposed EAV-Net
from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"


dataset_name = "avdrive" # 

if dataset_name == "avdrive":
    train_data_path = "./DRIVE_AV/train/"
    valid_data_path = "./DRIVE_AV/test/"
    N_epochs = 2000
    lr_decay_step = [1500]
    lr_init = 0.001
    batch_size = 16
    test_epoch = 10

def train_net(net, device, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    print('Train images: %s' % len(train_loader.dataset))
    print('Valid  images: %s' % len(valid_loader.dataset))
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_loss = float('inf')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        net.train()
        train_loss = 0
        with tqdm(total=train_loader.__len__()) as pbar:
            for i, (image, image2, label, label_vessel, filename) in enumerate(train_loader):
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                image2 = image2.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                output1, x13, x14, x15 = net(image)
                output2, x23, x24, x25 = net(image2)
                prob1 = torch.softmax(output1, dim=1)
                prob2 = torch.softmax(output2, dim=1)+1e-7

                f13 = torch.softmax(x13, dim=1)
                f14 = torch.softmax(x14, dim=1)
                f15 = torch.softmax(x15, dim=1)

                f23 = torch.softmax(x23, dim=1) + 1e-7
                f24 = torch.softmax(x24, dim=1) + 1e-7
                f25 = torch.softmax(x25, dim=1) + 1e-7

                inv_loss = F.kl_div(prob2.log(), prob1, reduction='mean')
                seg_loss = criterion(output1, label.long())

                f3_kl_loss = F.kl_div(f23.log(), f13, reduction='mean')
                f4_kl_loss = F.kl_div(f24.log(), f14, reduction='mean')
                f5_kl_loss = F.kl_div(f25.log(), f15, reduction='mean')

                total_loss = seg_loss + inv_loss + f3_kl_loss + f4_kl_loss + f5_kl_loss
                total_loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=float(total_loss.cpu()), epoch=epoch)
                pbar.update(1)

        # Validation
        if ((epoch+1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, image2, label, label_vessel, filename) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred_av, predf3, predf4, predf5 = net(image)
                loss = criterion(pred_av, label.long())
                val_loss = val_loss + loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), './snapshot/drive_best.pth')
                print('saving model............................................')
            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()

if __name__ == "__main__":
    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=3)
    net.to(device=device)
    train_net(net, device)
