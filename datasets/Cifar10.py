import numpy as np
import os
from constants import CIFAR10_DATADIR


class Dataset:
    def __init__(self, imgs_path_train, labels_path_train, imgs_path_test, labels_path_test,
                 name, num_classes, num_train, num_test):
        self.imgs_path_train = imgs_path_train
        self.labels_path_train = labels_path_train
        self.imgs_path_test = imgs_path_test
        self.labels_path_test = labels_path_test
        self.name = name
        self.num_classes = num_classes
        self.num_train = num_train
        self.num_test = num_test

    def format_labels(self, labels):
        import tensorflow as tf
        slim = tf.contrib.slim

        return slim.one_hot_encoding(labels, self.num_classes)

    def get_data_train(self):
        imgs = np.load(self.imgs_path_train)
        labels = np.load(self.labels_path_train)
        return imgs, labels

    def get_data_test(self):
        imgs = np.load(self.imgs_path_test)
        labels = np.load(self.labels_path_test)
        return imgs, labels


class CIFAR10(Dataset):
    CIFAR_TRAIN_IMGS = os.path.join(CIFAR10_DATADIR, "train_imgs.npy")
    CIFAR_TRAIN_LABELS = os.path.join(CIFAR10_DATADIR, "train_labels.npy")
    CIFAR_TEST_IMGS = os.path.join(CIFAR10_DATADIR, "test_imgs.npy")
    CIFAR_TEST_LABELS = os.path.join(CIFAR10_DATADIR, "test_labels.npy")

    def __init__(self, imgs_path_train=CIFAR_TRAIN_IMGS, labels_path_train=CIFAR_TRAIN_LABELS,
                 imgs_path_test=CIFAR_TEST_IMGS, labels_path_test=CIFAR_TEST_LABELS):
        Dataset.__init__(self, imgs_path_train, labels_path_train, imgs_path_test, labels_path_test,
                         'CIFAR-10', 10, 50000, 10000)
