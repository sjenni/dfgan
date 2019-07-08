from datasets.Cifar10 import CIFAR10
from models.GAN import GAN
from trainer.GanTrainer import GANTrainer
from utils import write_results

tag = 'standard_GAN'


def train_test(target_shape, dataset):
    model = GAN(batch_size=64, target_shape=target_shape, tag=tag)
    trainer = GANTrainer(model=model, dataset=dataset,
                         num_train_steps=100000, lr=0.0001)
    trainer.train_model(None)
    model.batch_size = 100
    fid, i_s = trainer.test_gan_all()
    write_results(fid, i_s, model, dataset, tag)


# CIFAR
target_shape = [32, 32, 3]
dataset = CIFAR10()
train_test(target_shape, dataset)
