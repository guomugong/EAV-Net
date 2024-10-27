# Efficient Retinal Artery/Vein Classification with Dense Color-Invariant Feature Learning
Please read our [paper](https://doi.org/10.1007/s00521-024-10696-z) for more details!

## Introduction:
Automatic retinal arteries and veins (A/V) classification is crucial in assisting clinicians in diagnosing cardiovascular and eye diseases. Deep learning models have been widely employed for A/V classification and have achieved remarkable performance. However, there are two primary challenges that need to be addressed: vessel discontinuity and A/V confusion.
To address the vessel discontinuity challenge, we have designed a multi-scale convolution block (MCB) that combines square convolution and strip convolution. This design aims to enhance the segmentation of tiny vessels by capturing more discriminative vessel features, thus effectively addressing the issue of vessel discontinuity.
Meantime, to address the A/V confusion challenge, we propose a Dense Color Invariant Feature Learning (DCIFL) method to enhance the model's robustness to color changes in retinal images.
Specifically, DCIFL uses KL divergences to align the distributions of both latent representations and predication maps of the raw input image and its color-transformed image.
Integrating MCB and DCIFL, we introduce EAV-Net, an efficient model for retinal A/V classification.
The proposed method has been tested on three publicly available datasets with an accuracy of 96.59\%, 97.28\%, and 99.34\% across AV-DRIVE, WIDE, and HRF datasets, respectively. These results demonstrate the superiority of our proposed approach in outperforming the state-of-the-art methods.

## Usage
1) Training EAV-Net on the AV-DRIVE dataset
```
python3 train.py
```
2) Testing
```
python3 predict.py
```

3) Evaluation
```
cd evaluation && python3 eval.py
```
## Usage
Pretrained checkpoint on AV-DRIVE dataset: snapshot/drive_best.pth
## License
This code can be utilized for academic research.
