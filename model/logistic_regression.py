import numpy as np


def initialize(dim):
    """
    Initialize w and b parameters to a (dim,1) vector and real 0. respectively

    Keyword arguments:
    dim -- the feature dimension of w

    Return
    w -- A (dim, 1) vector initialized with all zeros
    b --- 0.
    """
    w = np.zeros((dim, 1))
    return [w, 0.]


def sigmoid(x):
    """
    Apply the sigmoid function to x. X can be either a number or a matrix. If x is a matrix, then the sigmoid function
    is applied to all elements of x

    Keyword arguments:
    x -- the input to be applied sigmoid function to. Can be either a number or a matrix

    Return
    -- The sigmoid of x, which is sigmoid(x) if x is a number, or if x is a matrix is a new matrix where the sigmoid is
    applied to all the input elements of x
    """
    return 1. / (1. + np.exp((-x)))


def propagate(x, y, w, b, learning_rate):
    """
    Apply one propagate step of the Logistic Regression gradient descent algorithm. Since this implementation is done
    with a Neural Network mindset, one propagation step will consist of:
    - a forward propagation: compute the output of the function and apply the activation
    - a backward propagation: compute partial derivatives of the parameters

    After computing partial derivatives, then parametes w and b are updated with the negative of the derivatives
    multiplied by the learning rate, according to the gradient descent algorithm

    Keyword arguments:
    x -- A (n, m) matrix, where each column of the matrix is a single train example represented as (n, 1) vector
    y -- A (1, m) vector, where each element contains the right prediction (0|1) of each training example
    w -- A (n, 1) vector representing the logistic regression parameter w
    b -- A real number representing the logistic regression parameter b
    learning_rate -- A real number representing the learning rate of the gradient descent algorithm


    Return
    A dictionary containing the following elements:
    w -- the (n,1) vector logistic regression parameter w updated after one step of gradient descent
    b -- the real number logistic regression parameter b updated after one step of gradient descent
    cost -- the overall cost function after one step of gradient descent
    """
    assert x.shape[0] >= 1 and x.shape[1] >= 1
    assert x.shape[0] == w.shape[0]
    assert x.shape[1] == y.shape[1]
    assert y.shape[0] == w.shape[1] == 1
    m = x.shape[1]
    a = sigmoid(np.dot(w.T, x) + b)
    dz = a - y
    dw = np.dot(x, dz.T) / m
    db = np.sum(dz) / m
    w = w - learning_rate * dw
    b = b - learning_rate * db
    cost = np.squeeze(np.sum(y * np.log(a) + (1. - y) * np.log(1 - a)) / (-m))
    return {"w": w, "b": b, "cost": cost}


def predict(x, w, b, threshold):
    """
    Predict the output of the trained logistic regression model with parameter w and b, using the input threshold.
    The function will apply the logistic regression activation function, and then will output a 0|1 class based on the
    input threshold

    Keyword arguments:
    x -- A (n, m) matrix where each column of the matrix is a new example to be predicted with n features
    w -- The logistic regression (1,n) parameter vector
    b -- The logistic regression real number parameter b
    threshold -- Real number in [0,1] to understand the final output class (0|1) for each new example

    Return
    output_prediction -- A (1, m) vector, where each element is either 0 or 1 representing the predicted class of the
    model
    """
    assert len(x.shape) == 2
    assert x.shape[0] >= 1 and x.shape[1] >= 1
    assert len(w.shape) == 2
    assert w.shape[0] == x.shape[0] and w.shape[1] == 1
    assert 0 <= threshold <= 1

    m = x.shape[1]
    output_prediction = np.zeros((1, m))

    a = sigmoid(np.dot(w.T, x) + b)

    for i in range(m):
        output_prediction[0][i] = 0 if a[0][i] <= threshold else 1

    return output_prediction


def train(x, y, learning_rate, iteration, print_cost=False):
    """
    Train the Logistic Regression model given training examples = x, output classes for each training example = y,
    for a given number of iterations. The train will be performed with a gradient descent algorithm with a Neural
    Network mindset

    Keyword arguments:
    x -- A (n, m) matrix, where each column of the matrix is a single train example with n features
    y -- A (1, m) vector, where each element contains the right prediction (0|1) of each training example
    learning_rate -- A real number representing the learning rate of the gradient descent algorithm
    iteration -- An integer representing the number of iterations to be performed for the gradient descent algorithm
    print_cost -- If set to True, print the cost function value every 100th iterations

    Return
    costs -- A list containing each value of the cost function every 100th iterations
    A dictionary containing:
    w -- Logistic Regression trained parameter w
    b -- Logistic Regression trained parameter b
    """
    assert len(x.shape) == 2
    assert x.shape[0] >= 1 and x.shape[1] >= 1
    assert len(y.shape) == 2
    assert y.shape[1] == x.shape[1]
    assert y.shape[0] == 1
    assert learning_rate > 0
    assert iteration >= 0

    w, b = initialize(x.shape[0])
    costs = []

    for i in range(iteration):
        intermediate = propagate(x, y, w, b, learning_rate)

        w = intermediate["w"]
        b = intermediate["b"]
        cost = intermediate["cost"]

        # Print every #100th iteration
        if print_cost and i % 100 == 0:
            costs.append(cost)
            print("Training iteration number {} (out of {}), cost function value: {}".format(i, iteration, cost))

    return {"w": w, "b": b}, costs


def model(x_train, y_train, x_test, y_test, iteration, learning_rate, threshold, print_cost=False):
    """
    Build and train the Logistic Regression model with a Neural Network implementation.
    The train will be performed with a gradient descent algorithm performing a forward propagation step
    and a backward propagation step, like in a generic Neural Network model

    Keyword arguments:
    x_train -- A (n, m) matrix, where each column of the matrix is a single train example with n features. Each element
               of the matrix is expected to be a feature real value in [0,1]
    y_train -- A (1, m) vector, where each element contains the right prediction (0|1) of each training example
    x_test -- Similarly to x_training, A (n, m) matrix that will be used for testing the model rather than training
    y_test -- A (1, m) vector, where each element contains the right prediction (0|1) of each test example
    iteration -- An integer representing the number of iterations to be performed for the gradient descent algorithm
    learning_rate -- A real number representing the learning rate of the gradient descent algorithm
    threshold -- Real number in [0,1] to understand the final output class (0|1) for each new example
    print_cost -- If set to True, print the cost function value every 100th iterations of the gradient descent algorithm

    Return
    A dictionary containing the following data:
    learning_rate -- The learning rate used to train the model
    iteration -- Number of iterations used to train the model
    threshold -- Threshold used to compute the output predicted class given the model activation function
    costs -- A list containing the cost function value for every 100th iteration
    w -- The learned logistic regression parameters represent as (n,1) vector
    b -- The learned logistic regression real number parameter
    prediction_train -- A (m_train,1) vector containing the predicted value on the training set with the computed model
    prediction_test -- A (m_test,1) vector containing the predicted value on the test set with the computed model
    """
    assert len(x_train.shape) == len(y_train.shape) == len(x_test.shape) == len(y_test.shape) == 2
    assert x_train.shape[0] == x_test.shape[0] >= 1
    assert x_train.shape[1] == y_train.shape[1] >= 1
    assert x_test.shape[1] == y_test.shape[1]
    assert y_test.shape[0] == 1
    assert 0 <= learning_rate <= 1
    assert 0 <= threshold <= 1

    params, costs = train(x_train, y_train, learning_rate, iteration, print_cost)

    w, b = params["w"], params["b"]

    prediction_train = predict(x_train, w, b, threshold)
    prediction_test = predict(x_test, w, b, threshold)

    # Print train/test Errors
    if print_cost:
        print("Accuracy on train set: {} %".format(100 - np.mean(np.abs(prediction_train - y_train)) * 100))
        print("Accuracy on test set: {} %".format(100 - np.mean(np.abs(prediction_test - y_test)) * 100))

    return {"learning_rate": learning_rate,
            "iteration": iteration,
            "threshold": threshold,
            "costs": costs,
            "w": w,
            "b": b,
            "prediction_train": prediction_train,
            "prediction_test": prediction_test,
            }
