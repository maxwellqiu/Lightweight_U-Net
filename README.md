# Lightweight U-Net for Breast Ultrasound Image Segmentation

> Wenyang Qiu, Ethan Hamburg, Yisca Perez, Yinkun Zhou, Soorena Salari, and Yaser Esmaeili Salehani

Concordia University, Montréal, Canada  
Email: wenyang.qiu@concordia.ca


## Abstract  
Accurate segmentation of breast lesions in ultrasound images is essential for early breast cancer diagnosis and treatment planning. We propose a lightweight U-Net architecture that replaces standard convolutional encoder blocks with a MobileNetV2 backbone trained from scratch, reducing model complexity to 3.10 M parameters and 0.72 GFLOPs, while retaining rich feature extraction. Evaluated on the BUS-BRA dataset, our model outperformed the standard U-Net, Attention U-Net, and self-supervised learning models, achieving a Dice coefficient of 88.06\% and a mean Intersection over Union (mIoU) of 80.34\%, with an average inference latency of $1.44 \pm 0.41$ s per image. These results highlight the effectiveness of combining a lightweight encoder with attention mechanisms to achieve both high accuracy and computational efficiency in breast ultrasound segmentation.

**Keywords**: Breast Cancer Segmentation, U-Net, Lightweight Networks, Ultrasound Imaging, and Medical Image Analysis.


## Requirements

Python 3.11.9

```{python}
python -m pip install kagglehub pandas numpy opencv-python matplotlib torch torchvision albumentations scikit-learn
```
