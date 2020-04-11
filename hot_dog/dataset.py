import numpy as np
from pathlib import Path
from PIL import Image


# All images in the dataset will be re-sized with the following pixel sizes
pixel_size = (100, 100)


def load_dataset(path_train_pos, path_train_neg, path_test_pos, path_test_neg):
    """
    Load from disk the training and test images examples.
    Each image is converted into a (1, 256*256*3) vector, and values of the vector are normalized
    with 256.

    Keyword arguments:
    path_train_pos -- A string path to the folder containing positive image training examples
    path_train_neg -- A string path to the folder containing negative image training examples
    path_train_pos -- A string path to the folder containing positive image test examples
    path_train_neg -- A string path to the folder containing negative image test examples

    Return
    x_train -- matrix containing training examples, where each column of the matrix is one example
    y_train -- a one line vector containing the label (0|1) for each train example
    x_test  -- matrix containing test examples, where each column of the matrix is one example
    y_test  -- a one line vector containing the label (0|1) for each test example
    """
    x_train_pos = read_images(path_train_pos)
    print("Read a total of {} positive train examples".format(x_train_pos.shape[1]))
    y_train_pos = np.ones((1, x_train_pos.shape[1]))
    x_train_neg = read_images(path_train_neg)
    print("Read a total of {} negative train examples".format(x_train_neg.shape[1]))
    y_train_neg = np.zeros((1, x_train_neg.shape[1]))
    y_train = np.concatenate((y_train_pos, y_train_neg), axis=1)
    x_train = np.concatenate((x_train_pos, x_train_neg), axis=1)
    x_test_pos = read_images(path_test_pos)
    print("Read a total of {} positive test examples".format(x_test_pos.shape[1]))
    y_test_pos = np.ones((1, x_test_pos.shape[1]))
    x_test_neg = read_images(path_test_neg)
    print("Read a total of {} negative test examples".format(x_test_neg.shape[1]))
    y_test_neg = np.zeros((1, x_test_neg.shape[1]))
    y_test = np.concatenate((y_test_pos, y_test_neg), axis=1)
    x_test = np.concatenate((x_test_pos, x_test_neg), axis=1)
    print("Read a total of {} train examples, {} test examples, with {} number of features".format(
          x_train.shape[1], x_test.shape[1], x_train_pos.shape[0]))
    return x_train, y_train, x_test, y_test


def read_images(path):
    path_list = list(Path(path).glob("*.jpg"))
    m = len(list(path_list))
    a_matrix = np.zeros((m, pixel_size[0] * pixel_size[1] * 3))
    for i in range(m):
        a_matrix[i] = read_one_image(path_list[i]).T
    return a_matrix.T


def read_one_image(path):
    """
    Load one image from disk and convert it to a single column vector.
    The image is converted into a (1, 256*256*3) vector, and values of the vector are normalized
    with 256.

    Keyword arguments:
    path -- A string path to the image file on Disk

    Return
    -- A (256 * 256 * 3, 1) vector representing the image features, with each element a real number in [0, 1]
    """
    image = Image.open(path).resize(pixel_size)
    return (np.array(image) / 255.).reshape((1, pixel_size[0] * pixel_size[1] * 3)).T
