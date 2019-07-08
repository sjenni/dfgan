import tensorflow as tf
import net


class DFGAN:
    def __init__(self, batch_size, target_shape, tag='default', n_param=1.):
        self.name = 'DFGAN_ln_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.n_param = n_param
        self.net = net

    def gen(self, input, reuse=None, training=True):
        model, _ = self.net.generator(input, final_size=self.im_shape[0], is_training=training,
                                      scope='generator', reuse=reuse)
        return tf.tanh(model)

    def gen_noise(self, input, reuse=None, training=True):
        model, _ = self.net.generator(input, depth=8, final_size=self.im_shape[0], is_training=training,
                                      scope='noise_generator', reuse=reuse)
        return 2. * tf.nn.tanh(model)

    def disc(self, input, reuse=None, training=True):
        model, _ = self.net.discriminator(input, is_training=training, scope='discriminator', reuse=reuse)
        return model

    def n_loss(self, scope, noise, preds_fake_n, preds_real_n):
        loss_real = -0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(preds_real_n),
                                                           preds_real_n, scope=scope)
        loss_fake = -0.5 * tf.losses.sigmoid_cross_entropy(tf.zeros_like(preds_fake_n),
                                                           preds_fake_n, scope=scope)

        mse_eps_loss = tf.losses.mean_squared_error(tf.zeros_like(noise), noise, scope=scope,
                                                    weights=self.n_param)

        tf.summary.scalar('losses/noise_fake_loss', loss_fake)
        tf.summary.scalar('losses/noise_real_loss', loss_real)
        tf.summary.scalar('losses/mse_eps_loss', mse_eps_loss)

        return loss_real + loss_fake + mse_eps_loss

    def g_loss(self, scope, preads_fake, preds_fake_n):
        loss_fake = 0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(preads_fake),
                                                          preads_fake,
                                                          scope=scope)
        loss_fake += 0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(preads_fake),
                                                           preds_fake_n,
                                                           scope=scope)

        tf.summary.scalar('losses/generator_fake_loss', loss_fake)

        return loss_fake

    def d_loss(self, scope, preds_fake, preds_real, preds_fake_n, preds_real_n):
        loss_real = 0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(preds_real),
                                                          preds_real, scope=scope)
        loss_real += 0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(preds_real),
                                                           preds_real_n, scope=scope)
        loss_fake = 0.5 * tf.losses.sigmoid_cross_entropy(tf.zeros_like(preds_fake),
                                                          preds_fake,
                                                          scope=scope)
        loss_fake += 0.5 * tf.losses.sigmoid_cross_entropy(tf.zeros_like(preds_fake_n),
                                                           preds_fake_n,
                                                           scope=scope)
        loss = loss_real + loss_fake

        tf.summary.scalar('losses/disc_fake_loss', loss_fake)
        tf.summary.scalar('losses/disc_real_loss', loss_real)
        tf.summary.scalar('losses/disc_total_loss', loss)
        return loss
