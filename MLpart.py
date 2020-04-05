import numpy as np
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow.keras.models import load_model
from keras import metrics
import csv
import random
from skimage import img_as_bool


IMG_WIDTH = 28
IMG_HEIGHT = 28
PIXEL_DIM = 1
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT, PIXEL_DIM)
CLASSIFICATION_TH = 0.7


class Recognize():
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path, custom_objects={"top_3_acc": top_3_acc})
        self.labels_to_definitions, self.definitions_to_labels = Recognize.get_model_labels(labels_path)

    def test_model(self, x_test, y_test):
        # score trained model using validation set
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('test loss:', scores[0])
        print('test accuracy:', scores[1])

    @staticmethod
    def get_model_labels(labels_path):
        with open(labels_path, mode='r') as infile:
            reader = csv.reader(infile)
            labels_to_definitions = {int(rows[0]): rows[1] for rows in reader}
        with open(labels_path, mode='r') as infile:
            reader = csv.reader(infile)
            definitions_to_labels = {rows[1]: int(rows[0]) for rows in reader}
        return labels_to_definitions, definitions_to_labels

    @staticmethod
    def fit_image_to_prediction_dim(x):
        if x.shape[0] != IMG_WIDTH or x.shape[1] != IMG_HEIGHT:
            # plt.imshow(x, cmap="gray")
            # plt.show()
            # x = resize(x, (IMG_WIDTH, IMG_HEIGHT))
            # plt.imshow(x, cmap="gray")
            # plt.show()

            x = resize(x, (IMG_WIDTH, IMG_HEIGHT))
            x[x > 0.1] *= 1.5
            x[x > 1] = 1
            plt.imshow(x, cmap="gray")
            plt.show()

            # TODO check other ways to resize, like:
            # large image is shape (1, 128, 128)
            # small image is shape (1, 64, 64)
            # input_size = x.shape[0]
            # output_size = IMG_WIDTH
            # bin_size = input_size // output_size
            # new_x = x.reshape((1, output_size, bin_size,
            #                                    output_size, bin_size)).max(4).max(2)
            # plt.imshow(new_x, cmap="gray")
            # plt.show()
        x = x.reshape(IMG_WIDTH, IMG_HEIGHT, PIXEL_DIM)
        x = np.array([x, ])
        return x

    def predict(self, x):
        x = Recognize.fit_image_to_prediction_dim(x)
        y = self.model.predict(x)[0]
        return y

    def predict_specific_label(self, x, specific_label):
        return self.predict(x)[specific_label]

    def classify_specific_definition(self, x, specific_definition):
        label_prediction = self.predict_specific_label(x, self.definitions_to_labels[specific_definition])
        print("is it %s? %s" % (specific_definition, label_prediction))
        return label_prediction >= CLASSIFICATION_TH

    def predict_max_label(self, x):
        predictions = self.predict(x)
        arg_max = np.argmax(predictions)
        return self.labels_to_definitions[arg_max], predictions[arg_max]

    def get_random_definition(self):
        # return "banana"
        random_label = random.randint(0, len(self.labels_to_definitions)-1)
        random_definition = self.labels_to_definitions[random_label]
        return random_definition


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def load_quickdraw_images(quickdraw_images_path, max_images=100):
    x = np.load(quickdraw_images_path)[:max_images] # load all images
    x = x.astype(np.float32) / 255. # scale images
    # reshape to 28x28x1 grayscale image
    x = x.reshape(x.shape[0], IMG_WIDTH, IMG_HEIGHT, PIXEL_DIM)
    return x


def test_prediction(rec_obj):
    load_from_quickdraw_bananas = True
    if load_from_quickdraw_bananas:
        cur_dir_path = pathlib.Path().absolute()
        bananas_rel_path = "dataset/full_numpy_bitmap_banana_example.npy"
        bananas_abs_path = os.path.join(cur_dir_path, bananas_rel_path)
        bananas = load_quickdraw_images(bananas_abs_path)
        rand_banana_spot = random.randint(0, bananas.shape[0])
        banana = bananas[rand_banana_spot]
    else:
        banana = Image.open("C:/Users/EladShoham/Desktop/banana.jpg").convert('L')
        banana = np.asarray(banana)
        banana = banana.astype(np.float32) / 255. # scale images

    banana_img = banana.squeeze()
    plt.imshow(banana_img, cmap='gray')
    plt.show()
    banana_prediction = rec_obj.predict_specific_label(banana, rec_obj.definitions_to_labels["banana"])
    print("banana prediction: %s" % (banana_prediction))
    max_label, max_prediction = rec_obj.predict_max_label(banana)
    print("max label: %s. prediction: %s" % (max_label, max_prediction))
    classify_specific_definition = rec_obj.classify_specific_definition(banana, "banana")
    print(classify_specific_definition)


def get_default_model():
    return get_20_10000_model()


def get_20_10000_model():
    cur_dir_path = pathlib.Path().absolute()
    model_rel_path = "models/20/10000/model.h5"
    labels_rel_path = "models/20/10000/labels.csv"
    model_abs_path = os.path.join(cur_dir_path, model_rel_path)
    labels_abs_path = os.path.join(cur_dir_path, labels_rel_path)
    rec_obj = Recognize(model_abs_path, labels_abs_path)
    return rec_obj


def get_345_5000_model():
    cur_dir_path = pathlib.Path().absolute()
    model_rel_path = "models/345/5000/model.h5"
    labels_rel_path = "models/345/5000/labels.csv"
    model_abs_path = os.path.join(cur_dir_path, model_rel_path)
    labels_abs_path = os.path.join(cur_dir_path, labels_rel_path)
    rec_obj = Recognize(model_abs_path, labels_abs_path)
    return rec_obj


def main():
    rec_obj = get_default_model()
    test_prediction(rec_obj)


if __name__ == '__main__':
    main()