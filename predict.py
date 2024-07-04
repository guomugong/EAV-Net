import numpy as np
import torch
import cv2
from tqdm import tqdm
import torch.nn as nn
from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
import copy
from sklearn.metrics import roc_auc_score
from utils.eval_metrics import perform_metrics
#from fvcore.nn import FlopCountAnalysis, parameter_count_table

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset_name="avdrive"
model_path='./snapshot/drive_best.pth'


if dataset_name == "avdrive":
    test_data_path = "./DRIVE_AV/test/"
    raw_height = 584
    raw_width =  565

save_path='./results/'
save_path1='./results/A/'
save_path2='./results/V/'
save_path3='./results/whole/'
save_path4='./results/whole_color/'
save_path5='./results/vessel/'

def save_results(pred, save_path, filename):
    pred_a = pred.cpu().numpy().astype(np.double)[0][1]
    pred_v = pred.cpu().numpy().astype(np.double)[0][2]

    pred_all = pred.cpu().argmax(dim=1)[0] # 0, 1, 2, 3, 4
    pred_all = pred_all.numpy().astype(np.double)

    pred_a = pred_a * 255
    pred_v = pred_v * 255

    pred_c1 = np.zeros((pred_all.shape[0], pred_all.shape[1]))
    pred_c2 = np.zeros((pred_all.shape[0], pred_all.shape[1]))
    pred_c3 = np.zeros((pred_all.shape[0], pred_all.shape[1]))

    # 1 is blue = v, 2 is red = a

    pred_c1[pred_all==2]=255
    pred_c3[pred_all==1]=255

    pred_color = np.stack((pred_c1, pred_c2, pred_c3),axis=-1)
    cv2.imwrite(save_path1 + filename[0] + '_a.png', pred_a)
    cv2.imwrite(save_path2 + filename[0] + '_v.png', pred_v)
    cv2.imwrite(save_path3 + filename[0] + '.png', pred_all)
    cv2.imwrite(save_path4 + filename[0] + '_unet_color.png', pred_color)

    #print(f'{filename[0]} done!')

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, dataset_name)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader))
        device = torch.device('cuda')
        net = UNet(n_channels=3, n_classes=3)

        net.to(device=device)
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        net.eval()
        for image, image2, label, vessel, filename in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            pred, f3, f4, f5 = net(image)

            # dim=1 xaingsu xiangjia = 1
            pred = torch.softmax(pred, dim=1)
            # Save Segmentation Maps
            pred  = pred[:,:,:raw_height,:raw_width]  
            label = label[:,:raw_height,:raw_width]
            save_results(pred, save_path, filename)
            print(f'{filename} done!')
