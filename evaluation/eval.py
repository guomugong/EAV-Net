import numpy as np
import os
from libtiff import TIFF
import cv2
from PIL import Image
from torchvision import transforms
import torchvision

pred_path = "../results/whole/"
labels_path = "../DRIVE_AV/test/label/"

image_num = 20

pred_Dir = os.listdir(pred_path)
pred_Dir.sort()
label_Dir = os.listdir(labels_path)
label_Dir.sort()

ACC = np.zeros((1, 20))
SEN = np.zeros((1, 20))
SPE = np.zeros((1, 20))

from sklearn.metrics import f1_score
dice_values = []
for i in range(len(pred_Dir)):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    pred_name = os.path.join(pred_path, pred_Dir[i])
    label_name = os.path.join(labels_path, label_Dir[i])
    pred = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

    pred_a = np.zeros((pred.shape[0], pred.shape[1]))
    pred_v = np.zeros((pred.shape[0], pred.shape[1]))
    pred_raw_size = np.zeros((label.shape[0], label.shape[1]))
    pred_a[pred == 2] = 1
    pred_v[pred == 1] = 1
    pred_a = cv2.resize(pred_a, (label.shape[1], label.shape[0]))
    pred_v = cv2.resize(pred_v, (label.shape[1], label.shape[0]))
    pred_raw_size[pred_a == 1] = 2
    pred_raw_size[pred_v == 1] = 1
    pred = pred_raw_size


    def calculate_multiclass_dice(predictions, targets, num_classes):
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        dice_coefficient = f1_score(targets_flat, predictions_flat, average='macro')
        return dice_coefficient

    num_classes = 3
    dice_one = calculate_multiclass_dice(pred, label, num_classes)
    dice_values.append(dice_one)


    for x in range(label.shape[0]):
        for y in range(label.shape[1]):

            if (pred[x, y] == 1).all() and (label[x, y] == 1).all():
                TP = TP + 1
            if (pred[x, y] == 1).all() and (label[x, y] == 2).all():
                FP = FP + 1
            if (pred[x, y] == 2).all() and (label[x, y] == 2).all():
                TN = TN + 1
            if (pred[x, y] == 2).all() and (label[x, y] == 1).all():
                FN = FN + 1
    acc_one = (TP + TN) / (TP + TN + FN + FP)
    spe_one = TN / (TN + FP)
    sen_one = TP / (TP + FN)
    print(f' i :{i} acc_one: {acc_one} spe_one:{spe_one} sen_one: {sen_one} dice_one:{ dice_one }')

    ACC[0, i] = (TP + TN) / (TP + TN + FN + FP)
    SPE[0, i] = TN / (TN + FP)
    SEN[0, i] = TP / (TP + FN)

average_dice = np.mean(dice_values)

print(f' accuracy: {np.mean(ACC)} specificity:{np.mean(SPE)} sensitivity: {np.mean(SEN)} dice:{average_dice}')
