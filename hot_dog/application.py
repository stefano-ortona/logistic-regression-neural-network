from model import predict
from model import model
from hot_dog import load_dataset
from hot_dog import read_one_image
from hot_dog import dataset
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    """
    Build and train a model for the hot-dog/not hot-dog application, and try the model on new images.
    
    First download the dataset from Kaggle https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog/data
    and save it in your chosen BASE_FOLDER. Change the variable value 'base_folder' to your folder
    
    Then modify the 'new_image_path' variable to give a path for a new image to be predicted either as hot-dog
    or not hot-dog
    
    Then run the application. The application will first build and train the Logistic Regression model with a Neural
    Network approach. Once the model is built and trained, the model is used to predict whether a new chosen image is
    actually a hot dog or not
    
    NOTE: training the model with standard parameters takes around ~60 seconds with 2500 iterations
    """
    # Define here your base folder containing training and test examples with hot dot/not hot dog images
    # NOTE: it must be an absolute path
    base_folder = "BASE_FOLDER/hot-dog-not-hot-dog/"

    # Load the dataset into matrix of features
    x_train, y_train, x_test, y_test = load_dataset(base_folder + "train/hot_dog", base_folder + "train/not_hot_dog",
                                                    base_folder + "test/hot_dog", base_folder + "test/not_hot_dog")

    # x_train, y_train, x_test, y_test = load_dataset_2()
    # Train the model
    # Feel free to modify parameters such as iteration, learning rate or threshold to check variations in performance
    m = model(x_train, y_train, x_test, y_test,
              iteration=2500, learning_rate=0.005, threshold=0.5, print_cost=True)

    # Use the newly trained model to predict whether a new image is hot dog or not
    # Define here path to the new image to be predicted
    new_image_path = base_folder + "test.jpg"
    # Load the image as a feature vector
    new_image = read_one_image(new_image_path)
    # Predict the class on the new image
    y_out = predict(new_image, m["w"], m["b"], 0.5)

    assert y_out.shape == (1, 1)
    out_class = "hot dog" if y_out[0][0] == 1 else "not hot-dog"

    # Predict and show new image
    print("Image '{}' is predicted as '{}'".format(new_image_path, out_class))
    img_size = dataset.pixel_size
    plt.imshow(new_image.reshape((img_size[0], img_size[1], 3)))
    plt.title("Image '{}' is predicted as '{}'".format(new_image_path, out_class))
    plt.show(block=True)

    # Show cost variation plot
    all_costs = np.squeeze(m['costs'])
    plt.plot(all_costs)
    plt.ylabel('Cost Function value')
    plt.xlabel('Number of iterations (x100)')
    plt.title("Cost function variation with learning rate = {}".format(str(m["learning_rate"])))
    plt.show()
