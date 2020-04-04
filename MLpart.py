import pathlib
import os
from keras.models import load_model
from keras import metrics
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_DIM = 1


class Recognize():
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path, custom_objects={"top_3_acc": top_3_acc})
        self.labels_to_objects, self.objects_to_labels = Recognize.get_model_labels(labels_path)

    def test_model(self, x_test, y_test):
        # score trained model using validation set
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('test loss:', scores[0])
        print('test accuracy:', scores[1])

    @staticmethod
    def get_model_labels(labels_path):
        with open(labels_path, mode='r') as infile:
            reader = csv.reader(infile)
            labels_to_objects = {int(rows[0]): rows[1] for rows in reader}
        with open(labels_path, mode='r') as infile:
            reader = csv.reader(infile)
            objects_to_labels = {rows[1]: int(rows[0]) for rows in reader}
        return labels_to_objects, objects_to_labels

    def predict(self, x):
        y = self.model.predict(np.array([x, ]))[0]
        return y

    def predict_specific_label(self, x, specific_label):
        return self.predict(x)[specific_label]

    def predict_max_label(self, x):
        predictions = self.predict(x)
        arg_max = np.argmax(predictions)
        return self.labels_to_objects[arg_max], predictions[arg_max]

    def get_random_object(self):
        random_label = random.randint(0, len(self.labels_to_objects))
        random_object = self.labels_to_objects[random_label]
        return random_object


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def load_quickdraw_images(quickdraw_images_path, max_images=100):
    x = np.load(quickdraw_images_path)[:max_images] # load all images
    x = x.astype(np.float32) / 255. # scale images
    # reshape to 28x28x1 grayscale image
    x = x.reshape(x.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_DIM)
    return x


def test_prediction(rec_obj):
    cur_dir_path = pathlib.Path().absolute()
    bananas_rel_path = "dataset/full_numpy_bitmap_banana_example.npy"
    bananas_abs_path = os.path.join(cur_dir_path, bananas_rel_path)
    bananas = load_quickdraw_images(bananas_abs_path)
    rand_banana_spot = random.randint(0, bananas.shape[0])
    banana = bananas[rand_banana_spot]
    banana_img = banana.squeeze()
    plt.imshow(banana_img, cmap='gray')
    plt.show()
    banana_prediction = rec_obj.predict_specific_label(banana, rec_obj.objects_to_labels["banana"])
    print("banana prediction: %s" % (banana_prediction))
    max_label, max_prediction = rec_obj.predict_max_label(banana)
    print("max label: %s. prediction: %s" % (max_label, max_prediction))


def main():
    cur_dir_path = pathlib.Path().absolute()
    model_rel_path = "models/20/10000/model.h5"
    labels_rel_path = "models/20/10000/labels.csv"
    model_abs_path = os.path.join(cur_dir_path, model_rel_path)
    labels_abs_path = os.path.join(cur_dir_path, labels_rel_path)
    rec_obj = Recognize(model_abs_path, labels_abs_path)
    # random_object = rec_obj.get_random_object()
    # print(random_object)
    test_prediction(rec_obj)


if __name__ == '__main__':
    main()