import tensorflow as tf
import net


class GAN:
    def __init__(self, batch_size, target_shape, tag='default'):
        self.name = 'GAN_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.net = net

    def gen(self, input, reuse=None, training=True):
        model, _ = self.net.generator(input, final_size=self.im_shape[0], is_training=training,
                                      scope='generator', reuse=reuse)
        return tf.tanh(model)

    def disc(self, input, reuse=None, training=True):
        model, _ = self.net.discriminator(input, is_training=training, scope='discriminator', reuse=reuse)
        return model

    def g_loss(self, scope, preads_fake):
        loss_fake = tf.losses.sigmoid_cross_entropy(tf.ones_like(preads_fake),
                                                    preads_fake,
                                                    scope=scope)

        tf.summary.scalar('losses/generator_fake_loss', loss_fake)

        return loss_fake

    def d_loss(self, scope, preds_fake, preds_real):
        loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(preds_real),
                                                    preds_real,
                                                    scope=scope)
        loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(preds_fake),
                                                    preds_fake,
                                                    scope=scope)
        loss = loss_real + loss_fake

        tf.summary.scalar('losses/disc_fake_loss', loss_fake)
        tf.summary.scalar('losses/disc_real_loss', loss_real)
        tf.summary.scalar('losses/disc_total_loss', loss)
        return loss

