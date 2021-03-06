import tensorflow as tf
from GanTrainer import GANTrainer
from tensorflow.python.ops import control_flow_ops
from utils import get_variables_to_train, montage_tf

slim = tf.contrib.slim


class SRGANTrainer(GANTrainer):
    def __init__(self, *args, **kwargs):
        GANTrainer.__init__(self, *args, **kwargs)

    def make_fake(self, img, noise):
        f_img = img+noise
        return f_img

    def build_generator(self, batch_queue, opt, scope):
        noise_samples = self.get_noise_sample()
        fake_imgs = self.model.gen(noise_samples)
        noise = self.model.gen_noise(noise_samples)

        # Create the model
        disc_input_fake = tf.concat([self.make_fake(fake_imgs, noise), fake_imgs], 0)

        preds_disc_fake = self.model.disc(disc_input_fake)

        preds_fake_n, preds_fake = tf.split(preds_disc_fake, 2, 0)

        # Compute losses
        loss_g = self.model.g_loss(scope, preds_fake, preds_fake_n)

        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss_g = control_flow_ops.with_dependencies([updates], loss_g)

        # Calculate the gradients for the batch of data on this tower.
        grads_g = opt.compute_gradients(loss_g, get_variables_to_train('generator'))

        self.summaries += tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss_g, grads_g, {}

    def build_discriminator(self, batch_queue, opt, scope):
        imgs_train, _ = batch_queue.get_next()
        imgs_train.set_shape([self.model.batch_size, ] + self.model.im_shape)

        noise_samples = self.get_noise_sample()
        fake_imgs = self.model.gen(noise_samples)
        noise = self.model.gen_noise(noise_samples)

        tf.summary.image('imgs/train', montage_tf(imgs_train, 4, 16), max_outputs=1)
        tf.summary.image('imgs/train_noisy', montage_tf(self.make_fake(imgs_train, noise), 4, 16), max_outputs=1)
        tf.summary.image('imgs/fake', montage_tf(fake_imgs, 4, 16), max_outputs=1)
        tf.summary.image('imgs/fake_noisy', montage_tf(self.make_fake(fake_imgs, noise), 4, 16), max_outputs=1)

        disc_input_fake = tf.concat([self.make_fake(fake_imgs, noise), fake_imgs], 0)
        disc_input_real = tf.concat([self.make_fake(imgs_train, noise), imgs_train], 0)

        preds_disc_fake = self.model.disc(disc_input_fake)
        preds_disc_real = self.model.disc(disc_input_real, reuse=True)

        preds_real_n, preds_real = tf.split(preds_disc_real, 2, 0)
        preds_fake_n, preds_fake = tf.split(preds_disc_fake, 2, 0)

        # Compute losses
        loss = self.model.d_loss(scope, preds_fake, preds_real, preds_fake_n, preds_real_n)
        loss_n = self.model.n_loss(scope, noise, preds_fake_n, preds_real_n)

        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)
            loss_n = control_flow_ops.with_dependencies([updates], loss_n)

        # Calculate the gradients for the batch of data on this tower.
        grads = opt.compute_gradients(loss, get_variables_to_train('discriminator'))
        grads_n = opt.compute_gradients(loss_n, get_variables_to_train('noise_generator'))

        self.summaries += tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss, grads+grads_n, {}
