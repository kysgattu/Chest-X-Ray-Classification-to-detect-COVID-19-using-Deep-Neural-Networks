# Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks
Millions of people have been impacted by the COVID-19 pandemic, which has resulted in severe morbidity and mortality. Chest radiography is one of the quick and accessible diagnostic techniques for detecting COVID-19. The project aims to develop a reliable method for detecting COVID-19 in patients by analyzing chest X-rays. Three Deep Learning systems based on LeNet, ResNet, and VGG-19 models are being developed for detecting COVID-19 in patients on a Chest Radiography dataset and GRADCAM will be applied to the best model to detect COVID-19-affected areas in the lungs.

This project aims to develop a quick and accurate diagnostic method using deep learning techniques to detect COVID-19 and other respiratory illnesses. We developed a deep-learning model that can classify chest X-ray images into four groups: COVID-19, viral pneumonia, non-COVID bacterial infection, and normal health by using three very popular deep-learning techniques to train the models. These models can be utilized in the detection of respiratory illnesses, especially COVID-19. The project also aims to create a visualization technique that helps identify the areas affected by the virus in the lungs of a patient and helps give proper treatment.


## Table of Contents
- [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Dataset Description](#dataset-description)
- [Model Implementation](#modules)
    - [Data Preprocessing](#dataprep)
    - [Developing Deep Learning Models](#model_dev)
        - [LeNet-5](#lenet)
        - [ResNet-101](#resnet)
        - [VGG-19](#vgg)
        - [DenseNet](#densenet)
      - [GRAD-CAM Visualisation](#gradcam)
      - [Results](#results)
- [Developers](#developers)
- [Links](#links)
- [References](#references)            

## Prerequisites <a name='prerequisites'></a>

### Environment <a name='environment'></a>

1. Python 3 Environment (Ancaonda preferred)
2. Python modules required:NumPy,Pandas, PIL, Scikit-learn, Keras, Tensorflow, Warnings, Display, Random, Opencv2,Matplotlib, Seaborn
3. Web Browser

OR
- Any Python3 IDE installed with above modules. (Pycharm is used in the development of this project)

### Dataset Description <a name='dataset-description'></a>

The models are developed using a data repository named COVID-19 Radiography dataset from Kaggle which is compiled by a team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia. The dataset consists of chest X-ray images for three pneumonia kind chest illnesses along with normal health. The dataset contains more than 42000 images divided into two folders one containing chest X-Ray images and the other containing their corresponding masks. For this project, we used only the Chest X-Ray images which contain 3616 COVID-19-positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images.

![alt tag](https://github.com/kysgattu/Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks/blob/main/Project-Images/DatasetDescription.png)

## Model Implementation<a name='modules'></a>

> ### Data Preprocessing <a name = 'dataprep'></a>

- The dataset provided in Kaggle contains both Chest X-Ray Images and also their corresponding masks. Our implementation does not use the mask images. Hence, these images are ignored.
- The images in the dataset are randomly divided into three different subsets – Training, Testing, and Validation sets with a ratio of 80-10-10 to help in training the model, evaluating the model, and finetuning the parameters during the training respectively. The images in the training and validation sets are subjected to Horizontal flip to accommodate the Data Augmentation. This helped in increasing the training parameters and help train the model in detecting minor variations in data. Then all the images are resized to the same shape of 224x224 size and the color of the images is fixed to a range of ‘RGB’. This helped the models get trained easier and prevent overfitting of the model. These preprocessed images are used for training the models.


Models are saved at: [Fully Trained Models](https://studentuml-my.sharepoint.com/:f:/g/personal/kamalyeshodharshastry_gattu_student_uml_edu/EpG7-B4JXkRMvd-j4QIEOR0B7rRU_Q-eFEKVLYuWtIavdg?e=yLlY11)
