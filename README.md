# SwissGermanToText: Swiss German to Text Translation with Deep Learning

**SwissGermanToText** is a deep learning project that translates spoken Swiss German into text. In this projject, the model is trained using a convolutional neural network (CNN) to recognize spoken Swiss German.

## Features

- **Speech-to-Text**: Translates spoken Swiss German into written text.
- **Deep Learning Model**: Uses a CNN architecture to recognize audio features and classify them into text.
- **Dataset**: The model is trained on a diverse dataset of male and female Swiss German speakers.

## Requirements

- Python 3.x
- Keras with TensorFlow backend
- Numpy, Scikit Learn, Matplotlib for data manipulation and visualization
- Librosa for audio processing

## Installation

  Clone the repository:
   ```bash
   git clone https://github.com/iamkrunalrk/Swiss-German-to-Text.git
   ```

   Run ```SwissGermanToText.ipynb``` file


## Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed for audio classification. The architecture is as follows:

- **Conv2D Layers**: Three convolutional layers with 32, 48, and 120 filters respectively, each followed by a ReLU activation function to learn spatial hierarchies in the audio data.
- **MaxPooling2D**: A pooling layer to reduce spatial dimensions and retain the most important features.
- **Dropout Layers**: Regularization layers that randomly drop units to prevent overfitting.
- **Dense Layers**: Fully connected layers that perform classification based on learned features.
- **Softmax Output**: The final output layer uses a softmax activation to classify the input into one of the possible classes (Swiss German digits).

Mathematically, the model learns to map audio features to class labels via a series of operations:

1. **Convolution Operation**: The convolution operation $\mathcal{C}(I, K)$ applied to an input image $I$ with a filter kernel $K$:

$$
\mathcal{C}(I, K) = \sum_{i,j} I(i,j) \cdot K(i,j)
$$
   This operation allows the network to detect spatial features in the audio spectrogram.

2. **Pooling**: The pooling operation reduces the dimensionality of the feature maps by selecting the maximum value from a region:
   
$$
P = \max(I(x, y)) \quad \text{for each window in } I
$$

3. **Dense Layer**: A fully connected layer computes the weighted sum of inputs $x$ and applies an activation function:

$$
y = f(Wx + b)
$$

   where $W$ is the weight matrix, $b$ is the bias, and $f$ is the activation function.

5. **Softmax Activation**: The output layer applies the softmax function to produce probabilities for each class:

$$
P(y=i|x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

   where  $z_i$  is the score for class  $i$ , and the denominator is the sum over all class scores.

## Performance

- **Training Accuracy**: ~92%
- **Test Accuracy**: ~90%

## Improvements

1. **Data Augmentation**: Added more diverse audio samples, including different voices (male and female).
2. **Model Tuning**: Adjusted parameters like batch size and feature dimensions to improve accuracy.

