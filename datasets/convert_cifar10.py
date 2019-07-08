import os

import cPickle
import numpy as np
import sys
import urllib
import tarfile

from constants import CIFAR10_DATADIR

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_SRC_DATADIR = os.path.join(CIFAR10_DATADIR, 'cifar-10-batches-py')


def _download_and_extract():
    dest_directory = CIFAR10_DATADIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\rDownloading %s %.2f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_batch(idx=0, load_test=False):
    if load_test:
        batch_file = os.path.join(CIFAR10_SRC_DATADIR, 'test_batch')
    else:
        batch_file = os.path.join(CIFAR10_SRC_DATADIR, 'data_batch_{}'.format(idx))
    with open(batch_file, 'rb') as fo:
        batch = cPickle.load(fo)
    imgs = batch['data']
    labels = batch['labels']
    return imgs, labels


def proc_imgs(imgs):
    imgs = imgs.astype(float)
    imgs /= 127.5
    imgs -= 1.
    imgs = imgs.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
    return imgs


def run():
    _download_and_extract()

    if not os.path.exists(CIFAR10_DATADIR):
        os.makedirs(CIFAR10_DATADIR)

    # Load training images
    imgs = []
    labels = []
    for i in range(5):
        imgs_, labels_ = load_batch(i+1)
        imgs.append(imgs_)
        labels += labels_

    imgs = np.concatenate(imgs)

    # Normalize the training images
    imgs = proc_imgs(imgs)
    np.save(os.path.join(CIFAR10_DATADIR, "train_imgs"), imgs)
    np.save(os.path.join(CIFAR10_DATADIR, "train_labels"), labels)

    # Load and normalize the test images
    imgs, labels = load_batch(load_test=True)
    imgs = proc_imgs(imgs)
    np.save(os.path.join(CIFAR10_DATADIR, "test_imgs"), imgs)
    np.save(os.path.join(CIFAR10_DATADIR, "test_labels"), labels)
