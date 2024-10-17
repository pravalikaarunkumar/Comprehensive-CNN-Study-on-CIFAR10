# CIFAR-10 Image Classification Using CNN Architectures

## Overview:
This project focuses on a comparative study of various Convolutional Neural Network (CNN) architectures for image classification on the CIFAR-10 dataset. The goal is to evaluate and understand the performance of popular CNN models such as Vanilla CNN, VGG-like, ResNet-like, LeNet, AlexNet, and DenseNet. The project was carried out using an **NVIDIA A100 GPU** to ensure efficient training and faster processing times, given the complexity of some of the architectures. The CIFAR-10 dataset, known for its simplicity yet significant challenge in computer vision tasks, contains 60,000 images across 10 classes, with 50,000 images for training and 10,000 images for testing. The images are small (32x32 pixels), making it an excellent dataset for experimenting with different architectures without requiring too much computational power.

## Dataset Description:
### CIFAR-10:
- **Number of Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Data**: 50,000 images
- **Test Data**: 10,000 images
- **Image Size**: 32x32 pixels, RGB format
The goal of the project is to classify the images into one of the 10 categories. Each image is a low-resolution (32x32) color image, and each class has an equal number of images, which makes the dataset balanced.

## Architectures Used:

### 1. **Vanilla CNN**
This is a simple convolutional neural network with basic layers to capture spatial hierarchies in the data. It consists of three convolutional layers followed by max-pooling and fully connected layers.
- **Layers**: 3 convolutional layers, followed by 2 max-pooling layers, and 1 fully connected layer.
- **Activation Function**: ReLU
- **Output Layer**: A fully connected layer with 10 output units representing the 10 classes.

### 2. **VGG Network**
The VGG-like network builds upon the idea of using deep, sequential convolutional layers. It was inspired by the VGG architecture, known for its simplicity and excellent performance in image classification tasks. This version includes six convolutional layers grouped in blocks.
- **Convolutional Blocks**: Two blocks with 64 filters, two with 128 filters, and two with 256 filters.
- **Pooling**: MaxPooling layers applied after every two convolutional layers.
- **Dense Layers**: Two dense layers with 512 units, followed by the output layer.

### 3. **ResNet**
The ResNet architecture introduced residual learning to CNNs, solving the vanishing gradient problem when training deeper networks. In this version, we use residual blocks that apply identity connections to allow the model to learn more efficiently.
- **Residual Blocks**: Four blocks with increasing numbers of filters (64, 128, 256, and 512).
- **Global Average Pooling**: Applied after the residual blocks, followed by the output layer.

### 4. **LeNet**
LeNet was one of the earliest CNN architectures, designed for digit recognition. It features a simple structure with two convolutional layers followed by average pooling and fully connected layers.
- **Convolutional Layers**: Two convolutional layers with 6 and 16 filters.
- **Pooling**: Average pooling layers.
- **Dense Layers**: Two fully connected layers with 120 and 84 units.

### 5. **AlexNet**
AlexNet is one of the pioneering CNN models that achieved groundbreaking results on the ImageNet dataset. It consists of five convolutional layers and uses dropout for regularization.
- **Convolutional Layers**: Five layers with varying filter sizes (64, 128, 256).
- **Dense Layers**: Two fully connected layers with 1024 units and dropout layers to prevent overfitting.

### 6. **DenseNet**
DenseNet is a more recent architecture that improves information flow between layers by connecting each layer to every other layer in a feed-forward fashion. DenseNet was particularly well-suited for the CIFAR-10 dataset, performing better due to its ability to use fewer parameters while maintaining efficiency.
- **Dense Blocks**: Three dense blocks with 12 layers each.
- **Transition Layers**: Used for downsampling the feature maps between the dense blocks.

## Training Configuration:
The following features and hyperparameters were used consistently across all models:
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Sparse Categorical Cross-Entropy (since CIFAR-10 is a multi-class classification problem)
- **Metrics**: Accuracy
- **Epochs**: 50 epochs were run for each model.
- **Batch Size**: 64

All models were trained using an **NVIDIA A100 GPU**, which provided the necessary computational power to handle the larger models (like ResNet and DenseNet) efficiently. The GPU accelerated the training process, especially during convolution and backpropagation steps, enabling quicker experimentation and tuning.

## Results and Evaluation
Each model was evaluated on the CIFAR-10 test set after training. The performance of the models is summarized below:
- **Vanilla CNN**: Achieved good performance as a basic model, but it lacked the depth required for more complex feature extraction.
- **VGG-like Network**: Performed significantly better than the Vanilla CNN due to its deeper architecture.
- **ResNet-like Network**: Showed excellent results, as expected, due to its ability to mitigate the vanishing gradient problem in deep networks.
- **LeNet**: While LeNet is an efficient and lightweight model, it was outperformed by more modern architectures.
- **AlexNet**: Provided strong results, particularly due to its ability to learn complex patterns in the data through deeper convolutional layers.
- **DenseNet**: **Achieved the best performance** among all the models. DenseNet's densely connected layers allowed it to learn features more efficiently, resulting in superior accuracy on the CIFAR-10 dataset.

## Conclusion
This project provided a comprehensive comparison of various CNN architectures for image classification using the CIFAR-10 dataset. The **DenseNet model** demonstrated the best performance due to its innovative use of dense connectivity. ResNet also performed well, confirming its robustness in handling deeper networks. The study highlights the importance of choosing the right architecture based on the complexity of the task and the computational resources available.

## Contributors:  
[Pravalika Arunkumar](https://github.com/pravalikaarunkumar)

--- 

