# 🧥 Fashion-MNIST Image Classification with CNN and Dropout

## 🧾 Overview

This project demonstrates how to use a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras** to classify images from the **Fashion-MNIST** dataset. The model is trained to recognize 10 categories of clothing using grayscale images. To enhance generalization, the model incorporates **Dropout** layers and **EarlyStopping** during training.

## 📦 Dataset

The **Fashion-MNIST** dataset contains:

- 60,000 training images
- 10,000 test images  
- Image size: 28x28 pixels, grayscale  
- Labels: 10 fashion item categories

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

Dataset reference: [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)

## 🛠 Technologies Used

- Python
- TensorFlow + Keras
- NumPy
- Matplotlib

## 🚀 Setup Instructions

1. **Clone the repository**

git clone <YOUR_REPO_URL>
cd <REPO_FOLDER>


2. Install dependencies

We recommend using a virtual environment.


pip install tensorflow matplotlib numpy
You can also use:


pip install -r requirements.txt
(Create requirements.txt using pip freeze > requirements.txt)

## 🧠 Model Architecture
This CNN is designed to balance performance and simplicity, using Dropout to prevent overfitting.

Conv2D: 32 filters, 3x3, ReLU

MaxPooling2D: 2x2

Conv2D: 64 filters, 3x3, ReLU

MaxPooling2D: 2x2

Dropout: 0.25

Flatten

Dense: 128 units, ReLU

Dropout: 0.5

Dense: 10 units, Softmax (for classification)

Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metric: Accuracy
Training Strategy: EarlyStopping with validation loss monitoring (patience=5)

## 📈 Results
Final Test Accuracy: ~[INSERT_YOUR_TEST_ACCURACY]

EarlyStopping prevented overfitting by restoring the best weights.

Visualizations include:

Accuracy & loss curves

Correct vs incorrect classification examples

Probability distributions for predictions

## 🖼 Visualizations
Plot of training & validation accuracy/loss over epochs

Sample predictions with confidence levels (correct in blue, incorrect in red)

## 🧪 How to Run
Open and run fashion_mnist_cnn.ipynb (or equivalent .py file).

This will:

Download the dataset

Preprocess and normalize images

Build and train the CNN

Evaluate and visualize performance

Alternatively, run from terminal:

python fashion_mnist_cnn.py
## 🔧 Potential Improvements
Add data augmentation for further robustness

Introduce Batch Normalization layers

Use learning rate schedulers

Try advanced architectures like ResNet or MobileNet

Deploy the model using Streamlit or Flask

## 👨‍💻 Author

Udit Singh Rawat

