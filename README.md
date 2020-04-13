

# Logistic Regression Neural Network
This repository contains a Neural Network implementation of 
[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) Binary Classifier.

The library implements APIs to train the model, and use the model to predict output classes on newly seen examples. The Logistic Regression model is suitable for binary classification tasks, where data has to be classified into two separate and distinct groups, such as tagging images that contain dogs or not. Despite the Neural Network approach, this model is not capable of reaching high quality accuracy results of complex Neural Networks with several hidden layers, as this model represents essentially a single layer Neural Network

## Prerequisites
The library is written in [Python3](https://www.python.org/), and requires [pip](https://pip.pypa.io/en/stable/) to install third party packages.

All the third party packages are listed in the [requirements.txt](requirements.txt) file. If you wish to use the model alone without the example application, then you only need to install Numpy. All other requirements are only needed to run the [example application](#example-application)

## Usage
Here we show how the use Logistic Regression model to train a new model and use the trained model to predict new classification in Python. The only requirement to be installed to use the model is [Numpy](https://numpy.org/) for matrix operations

### Train the Model
```python
from model.logistic_regression import model

trained_model = model(x_train, y_train, x_test, y_test, iteration=2000, learning_rate=0.005, threshold=0.5, print_cost=True)

print("Here is the prediction on the 10th test example: {}".format(trained_model["prediction_test"][0][9]))
```
The input of the function call is:

- _x_train_: a (n, m) numpy vector, where m is the number of training examples and n is the number of features. In this matrix each column is a single training example with n features
- _y_train_: a (1, m) numpy vector where each element is either 0 or 1, representing the truth classification for each training example
- _x_test_: a (n, m') numpy vector. Similar to _x_train_, n is the number of features and m' is the number of test examples. Each column in this matrix is one test example. This matrix can also be empty
- _y_test_: a (1, m') numpy vector where each element is either 0 or 1, representing the truth classification for each training example
- _iteration_: number of training iterations to be performed (integer)
- _learning_rate_: learning rate parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha"> (float)
- _threshold_: classification threshold (float <img src="https://render.githubusercontent.com/render/math?math=\in [0,1]"> )
- _print_cost_: if set to True, then the value of the cost function will be printed every 100th iterations

The output of the function is a dictionary containing the following elements:

- _learning_rate_: same as above
- _iteration_: same as above
- _threshold_: same as above
- _costs_: a list containing all cost function values computed during training every 100th iteration
- _w_: a (n,1) numpy vector, containing the values of the parameter w with n features
- _b_: real number parameter b
- _prediction_train_: a (1,m) vector where each element is either 0 or 1, representing the output classification of the model for each training example
- _prediction_test_: a (1,m') vector where each element is either 0 or 1, representing the output classification of the model for each test example

### Use the trained Model
After having trained the model and computed parameters w and b, you can use the model to make predictions on new data as follows:
```python
from model.logistic_regression import model
from model.logistic_regression import predict

trained_model = model(x_train, y_train, x_test, y_test, iteration=2000, learning_rate=0.005, threshold=0.5, print_cost=True)

w = trained_model["w"]
b = trained_model["b"]

y_out = predict(x_new, w, b, threshold=0.5)
```

The input of the function call is:

- _x_new_: a (n, m'') numpy vector, where m'' is the number of new examples to be predicted and n is the number of features. In this matrix each column is a single new example with n features to be predicted
- _w_: the parameter w built after training the model
- _b_: the parameter b built after training the model
- _threshold_: classification threshold

The output of the call is a (1, m'') numpy vector where each element is either 0 or 1, representing the output classification of the model for each new example.

Have a look at the example application to see the model trained and used in full fashion

## Example Application
The repository contains an application of the model for the famous hot-hog vs. not hot-dog [app](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3).
Given an image, the model will be able to predict whether the image contains a hot-dog or not.

### Install Requirements
The application, other than Numpy, requires a few other libraries to convert images to vectors and to plot functions. You can install all the requirements by running:
```shell script
$ pip install -r requirements.txt
```
Best to do so in a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Run the application
Browse to [application.py](hot_dog/application.py) and change variable *base_folder* to the folder path where you downloaded the hot-dog [dataset](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog/data), and variable *new_image_path* to the file path of an image to be predicted either as hot-dog or not hot-dog.

After changing the variable values, run the following command:
```shell script
$ python3 -m hot_dog.application
```

This will run the application, which will first train the models on all the examples of the downloaded dataset, and then will use the model to predict whether the chosen image contains a hot-dog or not. At the end of the run, the application will first show the chosen image with the predicted value, and will also plot the variation of the cost function value every 100th training iteration.

You will see that the output model most likely has over-fitted the training data, with a ~99% accuracy on the training set and only ~52% accuracy on the test set -- but hey, this is just a simple Logistic Regression model, it cannot achieve same quality of a deep Neural Network :sweat_smile:


## Changelog

- TODO

## Contacts
Stefano Ortona - _stefano dot ortona at gmail.com_
