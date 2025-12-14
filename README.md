# DATA37000 Final Project – Animal Image Classification

This project explores multi-class image classification using convolutional neural networks (CNNs). The goal is to classify images into one of five animal categories using both a baseline CNN trained from scratch and an improved transfer-learning model based on ResNet18.

## Dataset
A subset of the **Open Images Dataset (V4)** was used, containing the following classes:
- Bird  
- Cat  
- Dog  
- Horse  
- Sheep  

To keep training computationally manageable, each class was capped at 300 images (with Sheep having fewer available samples). The dataset is **not included** in this repository but can be accessed here: can be accessed here: https://drive.google.com/drive/folders/1bpz5bGlmrmiwm5JrwpIQR_Q3jQTUJzmE?usp=sharing

## Models
### Baseline CNN
- Custom CNN trained from scratch
- Three convolutional blocks followed by fully connected layers
- Exhibits strong overfitting and poor generalization

### Improved Model (ResNet18)
- Pretrained ResNet18 (ImageNet)
- Frozen convolutional backbone
- Trained final classification layer only
- Includes data augmentation and normalization
- Achieves substantially better generalization

## Results
- **Baseline CNN**: ~28% validation accuracy with strong overfitting
- **ResNet18**: ~80% validation accuracy with stable training and strong per-class performance
