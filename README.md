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

> ### Developing Deep Learning Models <a name = 'model_dev'></a>
- We have trained three independent models – LeNet-5, ResNet-101, and VGG-16 using Keras-TensorFlow. These models are chosen for their well-known success with Image classification problems. One additional model is also developed to check the efficiency of DenseNet architecture on the problem.

#### LeNet-9 <a name = 'lenet'></a>
- LeNet is a convolutional neural network (CNN) proposed by Yann LeCun [7] in 1998 and was one of the first successful applications of CNNs. LeNet-5 is a simple architecture with seven layers of which the first two layers are convolutional layers, followed by two subsampling layers, and then two fully connected layers concluding with one output layer. During the training process, the weights of the model are adjusted to minimize the difference between the predicted class and the actual class of the chest X-ray image and are updated using the stochastic gradient descent (SGD) optimization algorithm.

![alt tag](https://github.com/kysgattu/Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks/blob/main/Project-Images/lenet_architecture.png)

- The model is initialized to train for a maximum of 20 epochs with categorical cross entropy loss function and an early stopping call back is applied to prevent overfitting of the model. The model stopped training at the 8th epoch as shown in image below and the best model is saved.

![alt tag](https://github.com/kysgattu/Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks/blob/main/Project-Images/lenet_training.png)

#### ResNet-101 <a name = 'resnet'></a>
- The ResNet model proposed in 2015 by Kaiming He et al. [8] is a deep and complex CNN architecture with residual connections that allows for the training of very deep networks. ResNet is made up of several residual blocks, each with two or three convolutional layers and a shortcut connection that bypasses them. The shortcut connection allows the input to be added to the residual block's output, thereby avoiding the vanishing gradient problem, and allowing the model to learn identity mappings.

![alt tag](https://github.com/kysgattu/Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks/blob/main/Project-Images/resnet_residual_blocks.png)

- We have used a variation of ResNet that is 101 layers deep including convolutional layers, batch normalization layers, and shortcut connections. The model is based on the Residual Network (ResNet) architecture, which was introduced in 2015. The Model we used is pre-trained on the ImageNet dataset consisting of one million images of 1000 categories which means it already knows how to detect important features in an image. We have added two new fully connected layers to the model using the Keras functional API. The first layer consists of 256 neurons with a ReLU activation function, followed by a final output layer with a softmax activation function those outputs probabilities for each of the categories. After adding the new layers, the pre-trained layers of the ResNet101 model are frozen, meaning that their weights will not be updated during training to prevent overfitting and to preserve the features that the pre-trained model has learned. The model is then compiled and trained with early stopping call back on validation loss, categorical crossentropy loss function, and Adam as the optimizer. The model is defined to be trained for a maximum of 20 epochs and the model stopped training at the 12th epoch as shown in Fig.3 and the best model is saved for further usage.

![alt tag](https://github.com/kysgattu/Chest-X-Ray-Classification-to-detect-COVID-19-using-Deep-Neural-Networks/blob/main/Project-Images/resnet_training.png)

Models are saved at: [Fully Trained Models](https://studentuml-my.sharepoint.com/:f:/g/personal/kamalyeshodharshastry_gattu_student_uml_edu/EpG7-B4JXkRMvd-j4QIEOR0B7rRU_Q-eFEKVLYuWtIavdg?e=yLlY11)
