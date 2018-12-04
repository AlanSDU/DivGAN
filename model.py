from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *


class DualNet(object):
    def __init__(self, sess, image_size_l=256, image_size_w=256, batch_size=1, fcn_filter_dim=64, \
                 A_channels=3, B_channels=3, dataset_name='facades', suffix='png', \
                 checkpoint_dir=None, lambda_A=20., lambda_B=20., ngf= 64, use_res=True, \
                 lambda_Sim_loss=20., sample_dir=None, loss_metric='L1', flip=False):
        self.df_dim = fcn_filter_dim
        self.flip = flip
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_Sim_loss = lambda_Sim_loss
        self.suffix = suffix
        self.ngf = ngf
        self.use_res = use_res

        self.sess = sess
        self.is_grayscale_A = (A_channels == 1)
        self.is_grayscale_B = (B_channels == 1)
        self.batch_size = batch_size
        self.image_size_l = image_size_l
        self.image_size_w = image_size_w
        self.fcn_filter_dim = fcn_filter_dim
        self.A_channels = A_channels
        self.B_channels = B_channels
        self.loss_metric = loss_metric

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        if use_res:
            res = "res"
        else:
            res = "unet"

        # directory name for output and logs saving
        self.dir_name = "%s-%s-img_sz_%sx%s-fltr_dim_%d-%s-lambda_AB_%s_%s_%s" % (
            self.dataset_name,
            res,
            self.image_size_l,
            self.image_size_w,
            self.fcn_filter_dim,
            self.loss_metric,
            self.lambda_A,
            self.lambda_B,
            self.lambda_Sim_loss
        )
        self.build_model()

    def build_model(self):
        ###    define place holders
        self.real_A_1 = tf.placeholder(tf.float32, [self.batch_size, self.image_size_l, self.image_size_w,
                                                    self.A_channels], name='real_A_1')
        self.real_B_1 = tf.placeholder(tf.float32, [self.batch_size, self.image_size_l, self.image_size_w,
                                                    self.B_channels], name='real_B_1')
        self.real_A_2 = tf.placeholder(tf.float32, [self.batch_size, self.image_size_l, self.image_size_w,
                                                    self.A_channels], name='real_A_2')
        self.real_B_2 = tf.placeholder(tf.float32, [self.batch_size, self.image_size_l, self.image_size_w,
                                                    self.B_channels], name='real_B_2')

        ###  define graphs
        self.real_A_1_fm, self.A_12B_fm_1, self.A_12B = self.A_g_net(self.real_A_1, reuseA=False, reuse1=False, name='GA1', use_res = True)
        self.real_A_2_fm, self.A_22B_fm_1, self.A_22B = self.A_g_net(self.real_A_2, reuseA=True, reuse1=False, name='GA2', use_res = True)
        self.real_B_1_fm, self.B_12A_fm_1, self.B_12A = self.B_g_net(self.real_B_1, reuseA=False, reuse1=False, name='GB1', use_res = True)
        self.real_B_2_fm, self.B_22A_fm_1, self.B_22A = self.B_g_net(self.real_B_2, reuseA=True, reuse1=False, name='GB2', use_res = True)
        self.A_12B_fm_2, self.A_12B2A_fm, self.A_12B2A = self.B_g_net(self.A_12B, reuseA=True, reuse1=True, name='GB1', use_res = True)
        self.A_22B_fm_2, self.A_22B2A_fm, self.A_22B2A = self.B_g_net(self.A_22B, reuseA=True, reuse1=True, name='GB2', use_res = True)
        self.B_12A_fm_2, self.B_12A2B_fm, self.B_12A2B = self.A_g_net(self.B_12A, reuseA=True, reuse1=True, name='GA1', use_res = True)
        self.B_22A_fm_2, self.B_22A2B_fm, self.B_22A2B = self.A_g_net(self.B_22A, reuseA=True, reuse1=True, name='GA2', use_res = True)

        if self.loss_metric == 'L1':
            self.A_loss = tf.reduce_mean(tf.abs(self.A_12B2A - self.real_A_1)) + tf.reduce_mean(
                tf.abs(self.A_22B2A - self.real_A_2))
            self.B_loss = tf.reduce_mean(tf.abs(self.B_12A2B - self.real_B_1)) + tf.reduce_mean(
                tf.abs(self.B_22A2B - self.real_B_2))
        elif self.loss_metric == 'L2':
            self.A_loss = tf.reduce_mean(tf.square(self.A_12B2A - self.real_A_1)) + tf.reduce_mean(
                tf.square(self.A_22B2A - self.real_A_2))
            self.B_loss = tf.reduce_mean(tf.square(self.B_12A2B - self.real_B_1)) + tf.reduce_mean(
                tf.square(self.B_22A2B - self.real_B_2))

        self.Ad_logits_fake1 = self.A_d_net(self.A_12B, reuseA=False, reuse1=False, name='DA1'),
        self.Ad_logits_real1 = self.A_d_net(self.real_B_1, reuseA=True, reuse1=True, name='DA1')
        self.Ad_logits_fake2 = self.A_d_net(self.A_22B, reuseA=True, reuse1=False, name='DA2')
        self.Ad_logits_real2 = self.A_d_net(self.real_B_2, reuseA=True, reuse1=True, name='DA2')
        self.Ad_loss_real = 0.5*celoss(self.Ad_logits_real1, tf.ones_like(self.Ad_logits_real1)) + \
                            0.5*celoss(self.Ad_logits_real2, tf.ones_like(self.Ad_logits_real2))
        self.Ad_loss_fake = 0.5*celoss(self.Ad_logits_fake1, tf.zeros_like(self.Ad_logits_fake1)) + \
                            0.5*celoss(self.Ad_logits_fake2, tf.zeros_like(self.Ad_logits_fake2))
        self.sim_loss_A_1_A_2 = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.real_A_1_fm - self.A_12B_fm_1, [-1]),
                                                                tf.reshape(self.real_A_2_fm - self.A_22B_fm_1, [-1])) /
                                                    tf.multiply(tf.norm(self.real_A_1_fm - self.A_12B_fm_1, ord=2),
                                                                tf.norm(self.real_A_2_fm - self.A_22B_fm_1, ord=2)))
        self.sim_loss_A_1_A_12B = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.real_A_1_fm - self.real_A_2_fm, [-1]),
                                                                  tf.reshape(self.A_12B_fm_1 - self.A_22B_fm_1, [-1])) /
                                                      tf.multiply(tf.norm(self.real_A_1_fm - self.real_A_2_fm, ord=2),
                                                                  tf.norm(self.A_12B_fm_1 - self.A_22B_fm_1, ord=2)))
        self.sim_loss_B_1_B_2 = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.real_B_1_fm - self.B_12A_fm_1, [-1]),
                                                                tf.reshape(self.real_B_2_fm - self.B_22A_fm_1, [-1])) /
                                                    tf.multiply(tf.norm(self.real_B_1_fm - self.B_12A_fm_1, ord=2),
                                                                tf.norm(self.real_B_2_fm - self.B_22A_fm_1, ord=2)))
        self.sim_loss_B_1_B_12A = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.real_B_1_fm - self.real_B_2_fm, [-1]),
                                                                  tf.reshape(self.B_12A_fm_1 - self.B_22A_fm_1, [-1])) /
                                                      tf.multiply(tf.norm(self.real_B_1_fm - self.real_B_2_fm, ord=2),
                                                                  tf.norm(self.B_12A_fm_1 - self.B_22A_fm_1, ord=2)))
        self.sim_loss_A_12B_A_22B = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.A_12B_fm_2 - self.A_12B2A_fm, [-1]),
                                                                    tf.reshape(self.A_22B_fm_2 - self.A_22B2A_fm, [-1])) /
                                                        tf.multiply(tf.norm(self.A_12B_fm_2 - self.A_12B2A_fm, ord=2),
                                                                    tf.norm(self.A_22B_fm_2 - self.A_22B2A_fm, ord=2)))
        self.sim_loss_A_12B_A_12B2A = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.A_12B_fm_2 - self.A_22B_fm_2, [-1]),
                                                                      tf.reshape(self.A_12B2A_fm - self.A_22B2A_fm,
                                                                                 [-1])) /
                                                          tf.multiply(tf.norm(self.A_12B_fm_2 - self.A_22B_fm_2, ord=2),
                                                                      tf.norm(self.A_12B2A_fm - self.A_22B2A_fm,
                                                                              ord=2)))
        self.sim_loss_B_12A_B_22A = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.B_12A_fm_2 - self.B_12A2B_fm, [-1]),
                                                                    tf.reshape(self.B_22A_fm_2 - self.B_22A2B_fm, [-1])) /
                                                        tf.multiply(tf.norm(self.B_12A_fm_2 - self.B_12A2B_fm, ord=2),
                                                                    tf.norm(self.B_22A_fm_2 - self.B_22A2B_fm, ord=2)))
        self.sim_loss_B_12A_B_12A2B = 0.5 * tf.reduce_sum(tf.multiply(tf.reshape(self.B_12A_fm_2 - self.B_22A_fm_2, [-1]),
                                                                      tf.reshape(self.B_12A2B_fm - self.B_22A2B_fm,
                                                                                 [-1])) /
                                                          tf.multiply(tf.norm(self.B_12A_fm_2 - self.B_22A_fm_2, ord=2),
                                                                      tf.norm(self.B_12A2B_fm - self.B_22A2B_fm,
                                                                              ord=2)))

        self.sim_loss = -0.125 * (self.sim_loss_A_1_A_2 +
                                  self.sim_loss_A_1_A_12B +
                                  self.sim_loss_B_1_B_2 +
                                  self.sim_loss_B_1_B_12A +
                                  self.sim_loss_A_12B_A_22B +
                                  self.sim_loss_A_12B_A_12B2A +
                                  self.sim_loss_B_12A_B_22A +
                                  self.sim_loss_B_12A_B_12A2B)

        self.Ad_loss = self.Ad_loss_fake + self.Ad_loss_real
        self.Ag_loss = 0.5*celoss(self.Ad_logits_fake1, labels=tf.ones_like(self.Ad_logits_fake1)) + \
                       0.5*celoss(self.Ad_logits_fake2, labels=tf.ones_like(self.Ad_logits_fake2)) + \
                       self.lambda_B * (self.B_loss) + self.lambda_Sim_loss * self.sim_loss

        self.Bd_logits_fake1 = self.B_d_net(self.B_12A, reuseA=False, reuse1=False, name='DB1')
        self.Bd_logits_real1 = self.B_d_net(self.real_A_1, reuseA=True, reuse1=True, name='DB1')
        self.Bd_logits_fake2 = self.B_d_net(self.B_22A, reuseA=True, reuse1=False, name='DB2')
        self.Bd_logits_real2 = self.B_d_net(self.real_A_2, reuseA=True, reuse1=True, name='DB2')
        self.Bd_loss_real = 0.5*celoss(self.Bd_logits_real1, tf.ones_like(self.Bd_logits_real1)) + \
                            0.5*celoss(self.Bd_logits_real2, tf.ones_like(self.Bd_logits_real2))
        self.Bd_loss_fake = 0.5*celoss(self.Bd_logits_fake1, tf.zeros_like(self.Bd_logits_fake1)) + \
                            0.5*celoss(self.Bd_logits_fake2, tf.zeros_like(self.Bd_logits_fake2))
        self.Bd_loss = self.Bd_loss_fake + self.Bd_loss_real
        self.Bg_loss = 0.5*celoss(self.Bd_logits_fake1, labels=tf.ones_like(self.Bd_logits_fake1)) + \
                       0.5*celoss(self.Bd_logits_fake2, labels=tf.ones_like(self.Bd_logits_fake2)) +\
                       self.lambda_A * (self.A_loss) + self.lambda_Sim_loss * self.sim_loss

        self.d_loss = self.Ad_loss + self.Bd_loss
        self.g_loss = self.Ag_loss + self.Bg_loss
        ## define trainable variables
        t_vars = tf.trainable_variables()
        self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
        self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]
        self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
        self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]
        self.d_vars = self.A_d_vars + self.B_d_vars
        self.g_vars = self.A_g_vars + self.B_g_vars
        self.saver = tf.train.Saver()

    def clip_trainable_vars(self, var_list):
        for var in var_list:
            self.sess.run(var.assign(tf.clip_by_value(var, -self.c, self.c)))

    def load_random_samples(self):
        # np.random.choice(
        sample_files = np.random.choice(glob(('./datasets/{}/val/A/*.' + self.suffix).format(self.dataset_name)),
                                        self.batch_size)
        sample_A_imgs_1 = [load_data(f, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=False) for
                           f in sample_files]

        sample_files = np.random.choice(glob(('./datasets/{}/val/A/*.' + self.suffix).format(self.dataset_name)),
                                        self.batch_size)
        sample_A_imgs_2 = [load_data(f, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=False) for
                           f in sample_files]

        sample_files = np.random.choice(glob(('./datasets/{}/val/B/*.' + self.suffix).format(self.dataset_name)),
                                        self.batch_size)
        sample_B_imgs_1 = [load_data(f, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=False) for
                           f in sample_files]

        sample_files = np.random.choice(glob(('./datasets/{}/val/B/*.' + self.suffix).format(self.dataset_name)),
                                        self.batch_size)
        sample_B_imgs_2 = [load_data(f, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=False) for
                           f in sample_files]

        sample_A_imgs_1 = np.reshape(np.array(sample_A_imgs_1).astype(np.float32),
                                     (self.batch_size, self.image_size_l, self.image_size_w, -1))
        sample_A_imgs_2 = np.reshape(np.array(sample_A_imgs_2).astype(np.float32),
                                     (self.batch_size, self.image_size_l, self.image_size_w, -1))
        sample_B_imgs_1 = np.reshape(np.array(sample_B_imgs_1).astype(np.float32),
                                     (self.batch_size, self.image_size_l, self.image_size_w, -1))
        sample_B_imgs_2 = np.reshape(np.array(sample_B_imgs_2).astype(np.float32),
                                     (self.batch_size, self.image_size_l, self.image_size_w, -1))

        return sample_A_imgs_1, sample_B_imgs_1, sample_A_imgs_2, sample_B_imgs_2

    def sample_shotcut(self, sample_dir, epoch_idx, batch_idx):
        sample_A_imgs_1, sample_B_imgs_1, sample_A_imgs_2, sample_B_imgs_2 = self.load_random_samples()

        Ag, A_12B2A_imgs, A_12B_imgs, A_22B2A_imgs, A_22B_imgs = self.sess.run(
            [self.sim_loss, self.A_12B2A, self.A_12B, self.A_22B2A, self.A_22B],
            feed_dict={self.real_A_1: sample_A_imgs_1,
                       self.real_A_2: sample_A_imgs_2,
                       self.real_B_1: sample_B_imgs_1,
                       self.real_B_2: sample_B_imgs_2})
        Bg, B_12A2B_imgs, B_12A_imgs, B_22A2B_imgs, B_22A_imgs = self.sess.run(
            [self.sim_loss, self.B_12A2B, self.B_12A, self.B_22A2B, self.B_22A],
            feed_dict={self.real_A_1: sample_A_imgs_1,
                       self.real_A_2: sample_A_imgs_2,
                       self.real_B_1: sample_B_imgs_1,
                       self.real_B_2: sample_B_imgs_2})

        save_images(A_12B_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_A_12B.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                          batch_idx))
        save_images(A_22B_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_A_22B.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                          batch_idx))
        save_images(A_12B2A_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_A_12B2A.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                            batch_idx))
        save_images(A_22B2A_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_A_22B2A.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                            batch_idx))

        save_images(B_12A_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_B_12A.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                          batch_idx))
        save_images(B_22A_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_B_22A.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                          batch_idx))
        save_images(B_12A2B_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_B_12A2B.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                            batch_idx))
        save_images(B_22A2B_imgs, [self.batch_size, 1],
                    ('./{}/{}/{:06d}_{:04d}_B_22A2B.' + self.suffix).format(sample_dir, self.dir_name, epoch_idx,
                                                                            batch_idx))

        print("[Sample] sim_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))

    def train(self, args):
        """Train Dual GAN"""
        decay = 0.9
        self.d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
            .minimize(self.d_loss, var_list=self.d_vars)

        self.g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
            .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter("./logs/" + self.dir_name, self.sess.graph)

        step = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...ignored...")
            print(" start training...")

        for epoch_idx in xrange(args.epoch):
            data_A = glob(('./datasets/{}/train/A/*.' + self.suffix).format(self.dataset_name))
            data_B = glob(('./datasets/{}/train/B/*.' + self.suffix).format(self.dataset_name))
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)
            epoch_size = min(len(data_A), len(data_B)) // (self.batch_size)
            print('[*] training data loaded successfully')
            print("#data_A: %d  #data_B:%d" % (len(data_A), len(data_B)))
            print('[*] run optimizor...')

            for batch_idx in xrange(0, (epoch_size - 1) // 2):
                imgA_batch_1 = self.load_training_imgs(data_A, batch_idx)
                imgA_batch_2 = self.load_training_imgs(data_A, batch_idx + epoch_size // 2 + 1)
                imgB_batch_1 = self.load_training_imgs(data_B, batch_idx)
                imgB_batch_2 = self.load_training_imgs(data_B, batch_idx + epoch_size // 2 + 1)

                print("Epoch: [%2d] [%4d/%4d]" % (epoch_idx, batch_idx, epoch_size))
                step = step + 1
                self.run_optim(imgA_batch_1, imgA_batch_2, imgB_batch_1, imgB_batch_2, step, start_time)
                ###
                if np.mod(step, 100) == 1:
                    self.sample_shotcut(args.sample_dir, epoch_idx, batch_idx)

                if np.mod(step, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, step)
                    ###

    def load_training_imgs(self, files, idx):
        batch_files = files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = [load_data(f, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=self.flip) for f
                      in batch_files]

        batch_imgs = np.reshape(np.array(batch_imgs).astype(np.float32),
                                (self.batch_size, self.image_size_l, self.image_size_w, -1))

        return batch_imgs

    def run_optim(self, batch_A_imgs_1, batch_A_imgs_2, batch_B_imgs_1, batch_B_imgs_2, counter, start_time):
        _, Adfake, Adreal, Bdfake, Bdreal, Ad, Bd = self.sess.run(
            [self.d_optim, self.Ad_loss_fake, self.Ad_loss_real, self.Bd_loss_fake, self.Bd_loss_real, self.Ad_loss,
             self.Bd_loss],
            feed_dict={self.real_A_1: batch_A_imgs_1, self.real_A_2: batch_A_imgs_2, self.real_B_1: batch_B_imgs_1,
                       self.real_B_2: batch_B_imgs_2})
        _, Ag, Bg, simloss = self.sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.sim_loss],
            feed_dict={self.real_A_1: batch_A_imgs_1, self.real_A_2: batch_A_imgs_2, self.real_B_1: batch_B_imgs_1,
                       self.real_B_2: batch_B_imgs_2})

        _, Ag, Bg, simloss = self.sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.sim_loss],
            feed_dict={self.real_A_1: batch_A_imgs_1, self.real_A_2: batch_A_imgs_2, self.real_B_1: batch_B_imgs_1,
                       self.real_B_2: batch_B_imgs_2})

        print("time: %4.4f, Ad: %.2f, Ag: %.2f, Bd: %.2f, Bg: %.2f,  Sim_loss: %.5f" \
              % (time.time() - start_time, Ad, Ag, Bd, Bg, simloss))
        print("Ad_fake: %.2f, Ad_real: %.2f, Bd_fake: %.2f, Bg_real: %.2f" % (Adfake, Adreal, Bdfake, Bdreal))

    def A_d_net(self, imgs1, y=None, reuseA=False, reuse1=False, name = 'A'):
        return self.discriminator(imgs1, prefix='A_d_', reuseA=reuseA, reuse1=reuse1, name0=name)

    def B_d_net(self, imgs1, y=None, reuseA=False, reuse1=False, name = 'B'):
        return self.discriminator(imgs1, prefix='B_d_', reuseA=reuseA, reuse1=reuse1, name0=name)


    def discriminator(self, image, y=None, prefix='A_d_', reuseA=False, reuse1=False, name0 = '1'):
        # image is 256 x 256 x (input_c_dim + output_c_dim)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA & reuse1:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            if '1' in name0:
                h0 = lrelu(dconv2d(image, self.df_dim, name=prefix + '1_h0_conv'))
            else:
                h0 = lrelu(dconv2d(image, self.df_dim, name=prefix + '2_h0_conv'))

            if reuseA:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(dconv2d(h0, self.df_dim * 2, name=prefix + 'h1_conv'), name=prefix + 'bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(dconv2d(h1, self.df_dim * 4, name=prefix + 'h2_conv'), name=prefix + 'bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(
                batch_norm(dconv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name=prefix + 'h3_conv'), name=prefix + 'bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = dconv2d(h3, 1, d_h=1, d_w=1, name=prefix + 'h4')

        return h4



    # def discriminator(self, images_1, images_2, y=None, prefix='A_d_', reuse=False):
    #     # image is 256 x 256 x (input_c_dim + output_c_dim)
    #     with tf.variable_scope(tf.get_variable_scope()) as scope:
    #         if reuse:
    #             scope.reuse_variables()
    #         else:
    #             assert scope.reuse == False
    #
    #         h01 = lrelu(conv2d(images_1, self.df_dim, name=prefix + 'h01_conv'))
    #         h02 = lrelu(conv2d(images_2, self.df_dim, name=prefix + 'h02_conv'))
    #         h0 = tf.concat(axis=0, values=[h01, h02], name=prefix + 'h0_conv')  # check the channel to cat
    #         # h0 is (128 x 128 x self.df_dim)
    #         h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name=prefix + 'h1_conv'), name=prefix + 'bn1'))
    #         # h1 is (64 x 64 x self.df_dim*2)
    #         h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name=prefix + 'h2_conv'), name=prefix + 'bn2'))
    #         # h2 is (32x 32 x self.df_dim*4)
    #         h3 = lrelu(
    #             batch_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name=prefix + 'h3_conv'), name=prefix + 'bn3'))
    #         # h3 is (32 x 32 x self.df_dim*8)
    #         h4 = conv2d(h3, 1, d_h=1, d_w=1, name=prefix + 'h4')
    #         return h4


    def fcn_resnet(self, image, reuseA=False, reuse1=False, prefix="A_g", name0="A"):

        if 'A' in prefix:
            output_c_dim = self.A_channels
        else:
            output_c_dim = self.B_channels
        with tf.variable_scope(name0):
            # image is 256 x 256 x input_c_dim
            if reuseA&reuse1:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            def residule_block(x, dim, ks=3, s=1, name='res'):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(cconv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(cconv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
                return y + x

            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            if '1' in name0:
                c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                c1 = tf.nn.relu(instance_norm(cconv2d(c0, self.ngf, 7, 1, padding='VALID', name=prefix + '1_e1_c'), name=prefix +'1_e1_bn'))
                c2 = tf.nn.relu(instance_norm(cconv2d(c1, self.ngf * 2, 3, 2, name=prefix + '1_e2_c'), name=prefix +'1_e2_bn'))
                c3 = tf.nn.relu(instance_norm(cconv2d(c2, self.ngf * 4, 3, 2, name=prefix + '1_e3_c'), name=prefix +'1_e3_bn'))
            else:
                c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                c1 = tf.nn.relu(instance_norm(cconv2d(c0, self.ngf, 7, 1, padding='VALID', name=prefix + '2_e1_c'), name=prefix +'2_e1_bn'))
                c2 = tf.nn.relu(instance_norm(cconv2d(c1, self.ngf * 2, 3, 2, name=prefix + '2_e2_c'), name=prefix +'2_e2_bn'))
                c3 = tf.nn.relu(instance_norm(cconv2d(c2, self.ngf * 4, 3, 2, name=prefix + '2_e3_c'), name=prefix +'2_e3_bn'))
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA:
                scope.reuse_variables()
            else:
                assert scope.reuse == False
            # define G network with 9 resnet blocks
            r1 = residule_block(c3, self.ngf * 4, name=prefix + 'r1')
            r2 = residule_block(r1, self.ngf * 4, name=prefix + 'r2')
            r3 = residule_block(r2, self.ngf * 4, name=prefix + 'r3')
            r4 = residule_block(r3, self.ngf * 4, name=prefix + 'r4')
            r5 = residule_block(r4, self.ngf * 4, name=prefix + 'r5')
            r6 = residule_block(r5, self.ngf * 4, name=prefix + 'r6')
            r7 = residule_block(r6, self.ngf * 4, name=prefix + 'r7')
            r8 = residule_block(r7, self.ngf * 4, name=prefix + 'r8')
            r9 = residule_block(r8, self.ngf * 4, name=prefix + 'r9')
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA & reuse1:
                scope.reuse_variables()
            else:
                assert scope.reuse == False
            if '1' in name0:
                d1 = cdeconv2d(r9, self.ngf * 2, 3, 2, name=prefix +'1_d1_dc')
                d1 = tf.nn.relu(instance_norm(d1, name=prefix +'1_d1_bn'))
                d2 = cdeconv2d(d1, self.ngf, 3, 2, name=prefix +'1_d2_dc')
                d2 = tf.nn.relu(instance_norm(d2, name=prefix +'1_d2_bn'))
                d2_fm = d2
                d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                pred = tf.nn.tanh(cconv2d(d2, output_c_dim, 7, 1, padding='VALID', name=prefix +'1_pred_c'))
            else:
                d1 = cdeconv2d(r9, self.ngf * 2, 3, 2, name=prefix + '2_d1_dc')
                d1 = tf.nn.relu(instance_norm(d1, name=prefix + '2_d1_bn'))
                d2 = cdeconv2d(d1, self.ngf, 3, 2, name=prefix + '2_d2_dc')
                d2 = tf.nn.relu(instance_norm(d2, name=prefix + '2_d2_bn'))
                d2_fm = d2
                d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                pred = tf.nn.tanh(cconv2d(d2, output_c_dim, 7, 1, padding='VALID', name=prefix + '2_pred_c'))
        return tf.nn.tanh(c1), tf.nn.tanh(d2_fm), pred


    def A_g_net(self, imgs, reuseA=False, reuse1=False, name='A', use_res = True):
        if use_res:
            return self.fcn_resnet(imgs, prefix='A_g_', reuseA=reuseA, reuse1=reuse1, name0=name)
        else:
            return self.fcn(imgs, prefix='A_g_', reuseA=reuseA, reuse1=reuse1, name0=name)

    def B_g_net(self, imgs, reuseA=False, reuse1=False, name='B', use_res = True):
        if use_res:
            return self.fcn_resnet(imgs, prefix='B_g_', reuseA=reuseA, reuse1=reuse1, name0=name)
        else:
            return self.fcn(imgs, prefix='B_g_', reuseA=reuseA, reuse1=reuse1, name0=name)

    def fcn(self, imgs, prefix=None, reuseA=False, reuse1=False, name0='A'):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA & reuse1:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            s_l = self.image_size_l
            s2_l, s4_l, s8_l, s16_l, s32_l, s64_l, s128_l = int(s_l / 2), int(s_l / 4), int(s_l / 8), int(
                s_l / 16), int(s_l / 32), int(s_l / 64), int(s_l / 128)

            s_w = self.image_size_w
            s2_w, s4_w, s8_w, s16_w, s32_w, s64_w, s128_w = int(s_w / 2), int(s_w / 4), int(s_w / 8), int(
                s_w / 16), int(s_w / 32), int(s_w / 64), int(s_w / 128)

            # imgs is (256 x 256 x input_c_dim)
            if '1' in name0:
                e1 = dconv2d(imgs, self.fcn_filter_dim, name=prefix + '1_e1_conv')
                # e1 is (128 x 128 x self.fcn_filter_dim)
                e2 = batch_norm(dconv2d(lrelu(e1), self.fcn_filter_dim * 2, name=prefix + '1_e2_conv'),
                                name=prefix + '1_bn_e2')
                # e2 is (64 x 64 x self.fcn_filter_dim*2)
                e3 = batch_norm(dconv2d(lrelu(e2), self.fcn_filter_dim * 4, name=prefix + '1_e3_conv'),
                                name=prefix + '1_bn_e3')
                # e3 is (32 x 32 x self.fcn_filter_dim*4)
            else:
                e1 = dconv2d(imgs, self.fcn_filter_dim, name=prefix + '2_e1_conv')
                # e1 is (128 x 128 x self.fcn_filter_dim)
                e2 = batch_norm(dconv2d(lrelu(e1), self.fcn_filter_dim * 2, name=prefix + '2_e2_conv'),
                                name=prefix + '2_bn_e2')
                # e2 is (64 x 64 x self.fcn_filter_dim*2)
                e3 = batch_norm(dconv2d(lrelu(e2), self.fcn_filter_dim * 4, name=prefix + '2_e3_conv'),
                                name=prefix + '2_bn_e3')
                # e3 is (32 x 32 x self.fcn_filter_dim*4)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            e4 = batch_norm(dconv2d(lrelu(e3), self.fcn_filter_dim * 8, name=prefix + 'e4_conv'), name=prefix + 'bn_e4')
            # e4 is (16 x 16 x self.fcn_filter_dim*8)
            e5 = batch_norm(dconv2d(lrelu(e4), self.fcn_filter_dim * 8, name=prefix + 'e5_conv'), name=prefix + 'bn_e5')
            # e5 is (8 x 8 x self.fcn_filter_dim*8)
            e6 = batch_norm(dconv2d(lrelu(e5), self.fcn_filter_dim * 8, name=prefix + 'e6_conv'), name=prefix + 'bn_e6')
            # e6 is (4 x 4 x self.fcn_filter_dim*8)
            e7 = batch_norm(dconv2d(lrelu(e6), self.fcn_filter_dim * 8, name=prefix + 'e7_conv'), name=prefix + 'bn_e7')
            # e7 is (2 x 2 x self.fcn_filter_dim*8)
            e8 = batch_norm(dconv2d(lrelu(e7), self.fcn_filter_dim * 8, name=prefix + 'e8_conv'), name=prefix + 'bn_e8')
            # e8 is (1 x 1 x self.fcn_filter_dim*8)

            self.d1, self.d1_w, self.d1_b = ddeconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128_l, s128_w, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd1', with_w=True)
            d1 = tf.nn.dropout(batch_norm(self.d1, name=prefix + 'bn_d1'), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = ddeconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64_l, s64_w, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd2', with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, name=prefix + 'bn_d2'), 0.5)

            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = ddeconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32_l, s32_w, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd3', with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, name=prefix + 'bn_d3'), 0.5)

            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = ddeconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16_l, s16_w, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd4', with_w=True)
            d4 = batch_norm(self.d4, name=prefix + 'bn_d4')

            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = ddeconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8_l, s8_w, self.fcn_filter_dim * 4],
                                                     name=prefix + 'd5', with_w=True)
            d5 = batch_norm(self.d5, name=prefix + 'bn_d5')
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuseA & reuse1:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            if prefix == 'B_g_':

                if '1' in name0:
                    self.d6, self.d6_w, self.d6_b = ddeconv2d(tf.nn.relu(d5),
                                                             [self.batch_size, s4_l, s4_w, self.fcn_filter_dim * 2],
                                                             name=prefix + '1_d6', with_w=True)
                    d6 = batch_norm(self.d6, name=prefix + '1_bn_d6')
                    d6 = tf.concat([d6, e2], 3)
                    # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

                    self.d7, self.d7_w, self.d7_b = ddeconv2d(tf.nn.relu(d6),
                                                             [self.batch_size, s2_l, s2_w, self.fcn_filter_dim],
                                                             name=prefix + '1_d7', with_w=True)
                    d7 = batch_norm(self.d7, name=prefix + '1_bn_d7')
                    d7 = tf.concat([d7, e1], 3)
                    # d7 is (128 x 128 x self.fcn_filter_dim*1*2)
                    self.d8, self.d8_w, self.d8_b = ddeconv2d(tf.nn.relu(d7), [self.batch_size, s_l, s_w, self.A_channels],
                                                         name=prefix + '1_d8', with_w=True)
                else:
                    self.d6, self.d6_w, self.d6_b = ddeconv2d(tf.nn.relu(d5),
                                                             [self.batch_size, s4_l, s4_w, self.fcn_filter_dim * 2],
                                                             name=prefix + '2_d6', with_w=True)
                    d6 = batch_norm(self.d6, name=prefix + '2_bn_d6')
                    d6 = tf.concat([d6, e2], 3)
                    # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

                    self.d7, self.d7_w, self.d7_b = ddeconv2d(tf.nn.relu(d6),
                                                             [self.batch_size, s2_l, s2_w, self.fcn_filter_dim],
                                                             name=prefix + '2_d7', with_w=True)
                    d7 = batch_norm(self.d7, name=prefix + '2_bn_d7')
                    d7 = tf.concat([d7, e1], 3)
                    # d7 is (128 x 128 x self.fcn_filter_dim*1*2)
                    self.d8, self.d8_w, self.d8_b = ddeconv2d(tf.nn.relu(d7), [self.batch_size, s_l, s_w, self.A_channels],
                                                         name=prefix + '2_d8', with_w=True)
            elif prefix == 'A_g_':
                if '1' in name0:
                    self.d6, self.d6_w, self.d6_b = ddeconv2d(tf.nn.relu(d5),
                                                             [self.batch_size, s4_l, s4_w, self.fcn_filter_dim * 2],
                                                             name=prefix + '1_d6', with_w=True)
                    d6 = batch_norm(self.d6, name=prefix + '1_bn_d6')
                    d6 = tf.concat([d6, e2], 3)
                    # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

                    self.d7, self.d7_w, self.d7_b = ddeconv2d(tf.nn.relu(d6),
                                                             [self.batch_size, s2_l, s2_w, self.fcn_filter_dim],
                                                             name=prefix + '1_d7', with_w=True)
                    d7 = batch_norm(self.d7, name=prefix + '1_bn_d7')
                    d7 = tf.concat([d7, e1], 3)
                    # d7 is (128 x 128 x self.fcn_filter_dim*1*2)
                    self.d8, self.d8_w, self.d8_b = ddeconv2d(tf.nn.relu(d7), [self.batch_size, s_l, s_w, self.B_channels],
                                                         name=prefix + '1_d8', with_w=True)
                else:
                    self.d6, self.d6_w, self.d6_b = ddeconv2d(tf.nn.relu(d5),
                                                             [self.batch_size, s4_l, s4_w, self.fcn_filter_dim * 2],
                                                             name=prefix + '2_d6', with_w=True)
                    d6 = batch_norm(self.d6, name=prefix + '2_bn_d6')
                    d6 = tf.concat([d6, e2], 3)
                    # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

                    self.d7, self.d7_w, self.d7_b = ddeconv2d(tf.nn.relu(d6),
                                                             [self.batch_size, s2_l, s2_w, self.fcn_filter_dim],
                                                             name=prefix + '2_d7', with_w=True)
                    d7 = batch_norm(self.d7, name=prefix + '2_bn_d7')
                    d7 = tf.concat([d7, e1], 3)
                    # d7 is (128 x 128 x self.fcn_filter_dim*1*2)
                    self.d8, self.d8_w, self.d8_b = ddeconv2d(tf.nn.relu(d7), [self.batch_size, s_l, s_w, self.B_channels],
                                                         name=prefix + '2_d8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(e1), tf.nn.tanh(self.d7), tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "DualNet.model"
        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test DualNet"""
        start_time = time.time()
        tf.global_variables_initializer().run()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
            test_dir = './{}/{}'.format(args.test_dir, self.dir_name)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            test_log = open(test_dir + 'evaluation.txt', 'a')
            test_log.write(self.dir_name)
            self.test_domain(args, test_log, type='A')
            self.test_domain(args, test_log, type='B')
            test_log.close()

    def test_domain(self, args, test_log, type='A'):
        test_files = glob(('./datasets/{}/val/{}/*.' + self.suffix).format(self.dataset_name, type))
        # load testing input
        print("Loading testing images ...")
        test_imgs = [
            load_data(f, is_test=True, image_size_l=self.image_size_l, image_size_w=self.image_size_w, flip=args.flip)
            for f in test_files]
        print("#images loaded: %d" % (len(test_imgs)))
        test_imgs = np.reshape(np.asarray(test_imgs).astype(np.float32),
                               (len(test_files), self.image_size_l, self.image_size_w, -1))
        test_imgs = [test_imgs[i * self.batch_size:(i + 1) * self.batch_size]
                     for i in xrange(0, len(test_imgs) // self.batch_size)]
        test_imgs = np.asarray(test_imgs)
        test_path = './{}/{}/'.format(args.test_dir, self.dir_name)
        # test input samples
        if type == 'A':
            for i in xrange(0, (len(test_files) - 1) // (2 * self.batch_size)):
                idx1 = i + 1
                idx2 = i + len(test_files) // (2 * self.batch_size) + 2
                filename_o1 = test_files[i * self.batch_size].split('/')[-1].split('.')[0]
                filename_o2 = test_files[(idx2 - 1) * self.batch_size].split('/')[-1].split('.')[0]
                print(filename_o1 + ' and ' + filename_o2)
                A_imgs_1 = np.reshape(np.array(test_imgs[i]),
                                      (self.batch_size, self.image_size_l, self.image_size_w, -1))
                A_imgs_2 = np.reshape(np.array(test_imgs[idx2 - 1]),
                                      (self.batch_size, self.image_size_l, self.image_size_w, -1))
                print("testing A image %d and %d" % (idx1, idx2))
                print(A_imgs_1.shape)
                A_12B_imgs, A_12B2A_imgs, A_22B_imgs, A_22B2A_imgs = self.sess.run(
                    [self.A_12B, self.A_12B2A, self.A_22B, self.A_22B2A],
                    feed_dict={self.real_A_1: A_imgs_1, self.real_A_2: A_imgs_2}
                )

                save_images(A_imgs_1, [self.batch_size, 1], test_path + filename_o1 + '_realA_1.' + self.suffix)
                save_images(A_imgs_2, [self.batch_size, 1], test_path + filename_o2 + '_realA_2.' + self.suffix)
                save_images(A_12B_imgs, [self.batch_size, 1], test_path + filename_o1 + '_A_12B.' + self.suffix)
                save_images(A_12B2A_imgs, [self.batch_size, 1], test_path + filename_o1 + '_A_12B2A.' + self.suffix)
                save_images(A_22B_imgs, [self.batch_size, 1], test_path + filename_o2 + '_A_22B.' + self.suffix)
                save_images(A_22B2A_imgs, [self.batch_size, 1], test_path + filename_o2 + '_A_22B2A.' + self.suffix)
        elif type == 'B':
            for i in xrange(0, (len(test_files) - 1) // (2 * self.batch_size)):
                idx1 = i + 1
                idx2 = i + len(test_files) // (2 * self.batch_size) + 2
                filename_o1 = test_files[i * self.batch_size].split('/')[-1].split('.')[0]
                filename_o2 = test_files[(idx2 - 1) * self.batch_size].split('/')[-1].split('.')[0]
                B_imgs_1 = np.reshape(np.array(test_imgs[i]),
                                      (self.batch_size, self.image_size_l, self.image_size_w, -1))
                B_imgs_2 = np.reshape(np.array(test_imgs[idx2 - 1]),
                                      (self.batch_size, self.image_size_l, self.image_size_w, -1))
                print("testing B image %d and %d" % (idx1, idx2))
                B_12A_imgs, B_12A2B_imgs, B_22A_imgs, B_22A2B_imgs = self.sess.run(
                    [self.B_12A, self.B_12A2B, self.B_22A, self.B_22A2B],
                    feed_dict={self.real_B_1: B_imgs_1, self.real_B_2: B_imgs_2}
                )
                save_images(B_imgs_1, [self.batch_size, 1], test_path + filename_o1 + '_realB_1.' + self.suffix)
                save_images(B_imgs_2, [self.batch_size, 1], test_path + filename_o2 + '_realB_2.' + self.suffix)
                save_images(B_12A_imgs, [self.batch_size, 1], test_path + filename_o1 + '_B_12A.' + self.suffix)
                save_images(B_22A_imgs, [self.batch_size, 1], test_path + filename_o2 + '_B_22A.' + self.suffix)
                save_images(B_12A2B_imgs, [self.batch_size, 1], test_path + filename_o1 + '_B_12A2B.' + self.suffix)
                save_images(B_22A2B_imgs, [self.batch_size, 1], test_path + filename_o2 + '_B_22A2B.' + self.suffix)
