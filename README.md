# Lightweight U-Net for Breast Ultrasound Image Segmentation

> Wenyang Qiu, Ethan Hamburg, Yisca Perez, Yinkun Zhou, Soorena Salari, and Yaser Esmaeili Salehani

Concordia University, Montréal, Canada  
Email: wenyang.qiu@concordia.ca


## Abstract  
The accurate segmentation of breast lesions in ultrasound images is crucial for early breast cancer diagnosis and treatment planning. We propose a lightweight U-Net that replaces standard convolutional encoder blocks with a pre-trained MobileNetV2 backbone, reducing the complexity to 2.40\, M parameters and 11.3\, GFLOPs ($\approx 1.13\times10^{10}$ FLOPs), while preserving rich feature extraction. Trained and evaluated on the BUS-BRA dataset, our model outperformed the standard U-Net, an Attention U-Net, and semi-supervised learning models, achieving a Dice coefficient of 89.39\% and mIoU of 82.21\% with an inference latency of $1.09 \pm 0.02$\,s per image. These results demonstrate that combining a lightweight encoder with attention mechanisms yields both efficiency and accuracy for breast ultrasound segmentation.

**Keywords**: Breast Cancer Segmentation, U-Net, Lightweight Networks, Ultrasound Imaging, and Medical Image Analysis.


## Requirements

Python 3.11.9

```{python}
python -m pip install kagglehub pandas numpy opencv-python matplotlib torch torchvision albumentations scikit-learn
```