# On Stabilizing Generative Adversarial Training with Noise [[Project Page]](https://sjenni.github.io/dfgan/) 

This repository contains demo code of our CVPR2019 [paper](https://arxiv.org/abs/1906.04612). It contains code for the training and evaluation of a DFGAN with learned noise on the CIFAR-10 dataset. 

## Requirements
The code is based on Python 2.7 and tensorflow 1.12.

## How to use it

### 1. Setup

- Set the paths to the data and log directories in **constants.py**.
- Run **init_datasets.py** to download and convert the CIFAR-10 dataset.

### 2. DFGAN training and evaluation 

- To train and evaluate a DFGAN with learned noise on CIFAR-10 run **run_DFGAN_ln.py**.
- To train and evaluate a standard GAN on CIFAR-10 run **run_standard_GAN.py**.
