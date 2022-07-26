import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(1)

import numpy as np


Pressure_train = np.load('../2D_array/train/pressure.npy')
Velocity_u_train = np.load('../2D_array/train/velocity_u.npy')
Velocity_v_train = np.load('../2D_array/train/velocity_v.npy')
T_train = np.load('../2D_array/train/t.npy')

Pressure_test = np.load('../2D_array/test/pressure.npy')
Velocity_u_test = np.load('../2D_array/test/velocity_u.npy')
Velocity_v_test = np.load('../2D_array/test/velocity_v.npy')
T_test = np.load('../2D_array/test/t.npy')

X = np.load('../2D_array/X.npy')[:, None]
Y = np.load('../2D_array/Y.npy')[:, None]


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = tf.sqrt(2. / (in_dim + out_dim))
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))


def gradient(Y, x):
    dummy = tf.ones_like(Y)
    grad = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(grad, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


def Navier_Stokes(p, u, v, t, x, y, Re):
    Y = tf.concat([p, u, v], axis=1)

    p = Y[:, 0:1]
    u = Y[:, 1:2]
    v = Y[:, 2:3]

    Y_t = gradient(Y, t)
    Y_x = gradient(Y, x)
    Y_y = gradient(Y, y)
    Y_x_x = gradient(Y_x, x)
    Y_y_y = gradient(Y_y, y)

    p_x = Y_x[:, 0:1]
    p_y = Y_y[:, 0:1]

    u_t = Y_t[:, 1:2]
    u_x = Y_x[:, 1:2]
    u_y = Y_y[:, 1:2]
    u_x_x = Y_x_x[:, 1:2]
    u_y_y = Y_y_y[:, 1:2]

    v_t = Y_t[:, 2:3]
    v_x = Y_x[:, 2:3]
    v_y = Y_y[:, 2:3]
    v_x_x = Y_x_x[:, 2:3]
    v_y_y = Y_y_y[:, 2:3]

    e_1 = u_t + u * u_x + v * u_y + p_x - 1 / Re * (u_x_x + u_y_y)
    e_2 = v_t + u * v_x + v * v_y + p_y - 1 / Re * (v_x_x + v_y_y)
    e_3 = u_x + v_y

    return e_1, e_2, e_3


def tf_Session():
    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def train_log(path, *inputs):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(inputs) + '\n')


class GAN:
    def __init__(self, t, x, y, p, u, v):
        self.t, self.x, self.y, self.p, self.u, self.v = t, x, y, p, u, v

        self.Reynolds = 100

        [self.t_tf, self.x_tf, self.y_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        [self.p_tf, self.u_tf, self.v_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        self.g_layers = [3] + 10 * [256] + [3]
        self.d_layers = [6] + 10 * [256] + [1]

        self.generator = Generator(self.g_layers)
        self.discriminator = Discriminator(self.d_layers)

        [self.g_p, self.g_u, self.g_v] = self.generator.nn(self.t_tf, self.x_tf, self.y_tf)

        self.d_real = self.discriminator.nn(self.t_tf, self.x_tf, self.y_tf, self.p_tf, self.u_tf, self.v_tf)
        self.d_fake = self.discriminator.nn(self.t_tf, self.x_tf, self.y_tf, self.g_p, self.g_u, self.g_v)

        self.saver = tf.train.Saver()
        self.model_path = '../models/gan/1 0.2 1_/gan{0}.ckpt'
        self.model_load_path = '../models/nonw'

        [self.e_1, self.e_2, self.e_3] = Navier_Stokes(self.g_p, self.g_u, self.g_v, self.t_tf, self.x_tf, self.y_tf,
                                                       self.Reynolds)

        self.epsilon = 1e-10

        self.d_loss = - tf.reduce_mean(tf.log(self.d_real + self.epsilon) + tf.log(1 - self.d_fake + self.epsilon))

        self.l2_Loss = mean_squared_error(self.g_p, self.p_tf) + \
                       mean_squared_error(self.g_u, self.u_tf) + \
                       mean_squared_error(self.g_v, self.v_tf)
        self.pde_residuals = mean_squared_error(self.e_1, 0.0) + \
                             mean_squared_error(self.e_2, 0.0) + \
                             mean_squared_error(self.e_3, 0.0)

        self.entropy_loss = -tf.reduce_mean(tf.log(self.d_fake + self.epsilon))
        self.g_loss = 1 * self.l2_Loss + 0.2 * self.pde_residuals + 1 * self.entropy_loss

        self.lr = 0.0001
        self.epochs = 30000
        self.batch_size = 10000

        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.d_loss)

        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.g_loss)

        self.session = tf_Session()

    def train(self):
        data_size = self.t.shape[0]

        running_time = 0
        start_time = time.time()
        log_save_path = '../results/gan/1 0.2 1_/loss.txt'

        for epoch in range(self.epochs):

            idx = np.random.choice(data_size, self.batch_size, replace=False)

            (t_data_batch, x_data_batch, y_data_batch) = (self.t[idx, :], self.x[idx, :], self.y[idx, :])

            (p_data_batch, u_data_batch, v_data_batch) = (self.p[idx, :], self.u[idx, :], self.v[idx, :])

            tf_dict = {
                self.t_tf: t_data_batch,
                self.x_tf: x_data_batch,
                self.y_tf: y_data_batch,
                self.p_tf: p_data_batch,
                self.u_tf: u_data_batch,
                self.v_tf: v_data_batch,
            }

            [_, g_loss, l2_Loss, entropy_loss, pde_residuals] = self.session.run([self.g_optimizer,
                                                                                                 self.g_loss,
                                                                                                 self.l2_Loss,
                                                                                                 self.entropy_loss,
                                                                                                 self.pde_residuals],
                                                                                                tf_dict)

            [_, d_loss] = self.session.run([self.d_optimizer, self.d_loss], tf_dict)

            if (epoch + 1) % 10 == 0:
                current_time = time.time()
                training_train = current_time - start_time
                running_time += training_train / 3600

                train_log(log_save_path, g_loss, l2_Loss, entropy_loss, pde_residuals, d_loss)

                print('epoch: %d  g_loss: %.4e  d_loss: %.4e  training time: %f s  running time: %f h'
                      % (epoch + 1, g_loss, d_loss, training_train, running_time))
                start_time = time.time()

            if (epoch + 1) % 10000 == 0:
                model_path = self.model_path.format(str(epoch + 1))
                save_path = self.saver.save(self.session, model_path)
                print('epoch: %d, models has been saved in file %s' % (epoch + 1, save_path))

    def predict(self, x, y, t):
        ckpt = tf.train.latest_checkpoint(self.model_load_path)
        self.saver.restore(self.session, ckpt)

        tf_dict = {self.x_tf: x,
                   self.y_tf: y,
                   self.t_tf: t}

        g_p = self.session.run(self.g_p, tf_dict)
        g_u = self.session.run(self.g_u, tf_dict)
        g_v = self.session.run(self.g_v, tf_dict)
        return g_p, g_u, g_v


class Generator:
    def __init__(self, layers):
        self.layers = layers
        self.layers_num = len(layers)

        self.W = []
        self.B = []
        self.G = []

        for i in range(self.layers_num - 1):
            w = tf.Variable(xavier_init([self.layers[i], self.layers[i + 1]]), dtype=tf.float32, name='weight')
            b = tf.Variable(tf.zeros(shape=[1, layers[i + 1]]), dtype=tf.float32, name='bias')
            g = tf.Variable(tf.ones(shape=[1, layers[i + 1]]), dtype=tf.float32, name='gamma')
            self.W.append(w)
            self.B.append(b)
            self.G.append(g)

    def nn(self, *inputs):
        H = tf.concat(inputs, axis=1)

        for i in range(self.layers_num - 1):
            #  weight normalization
            v = self.W[i] / tf.norm(self.W[i], axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, v)
            # add bias
            H = self.G[i] * H + self.B[i]
            # activation function
            if i < self.layers_num - 2:
                H = H * tf.sigmoid(H)
        outputs = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        return outputs


class Discriminator:
    def __init__(self, layers):
        self.layers = layers
        self.layers_num = len(layers)

        self.W = []
        self.B = []
        self.G = []

        for i in range(self.layers_num - 1):
            w = tf.Variable(xavier_init([self.layers[i], self.layers[i + 1]]), dtype=tf.float32, name='weight')
            b = tf.Variable(tf.zeros(shape=[1, layers[i + 1]]), dtype=tf.float32, name='bias')
            g = tf.Variable(tf.ones(shape=[1, layers[i + 1]]), dtype=tf.float32, name='gamma')
            self.W.append(w)
            self.B.append(b)
            self.G.append(g)

    def nn(self, *inputs):
        H = tf.concat(inputs, axis=1)

        for i in range(self.layers_num - 1):
            #  weight normalization
            v = self.W[i] / tf.norm(self.W[i], axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, v)
            # add bias
            H = self.G[i] * H + self.B[i]
            # activation function
            H = tf.sigmoid(H)
        outputs = H
        return outputs


if __name__ == '__main__':
    node_num = X.shape[0]
    time_step_train = T_train.shape[0]
    time_step_test = T_test.shape[0]

    T_train_t = np.tile(T_train, (1, node_num)).T
    T_test_t = np.tile(T_test, (1, node_num)).T
    X_t = np.tile(X, (1, time_step_train))
    Y_t = np.tile(Y, (1, time_step_train))

    t_train = T_train_t.flatten()[:, None]
    t_test = T_test_t.flatten()[:, None]
    x = X_t.flatten()[:, None]
    y = Y_t.flatten()[:, None]

    p_train = Pressure_train.flatten()[:, None]
    u_train = Velocity_u_train.flatten()[:, None]
    v_train = Velocity_v_train.flatten()[:, None]

    data_size = node_num * time_step_train

    idx = np.random.choice(data_size, data_size, replace=False)

    t_train_data = t_train[idx]
    x_train_data = x[idx]
    y_train_data = y[idx]

    p_train_data = p_train[idx]
    u_train_data = u_train[idx]
    v_train_data = v_train[idx]

    gan = GAN(t_train_data, x_train_data, y_train_data, p_train_data, u_train_data, v_train_data)

    # gan.train()

    for pre_id in range(time_step_test):
        test_t = T_test_t[:, pre_id].flatten()[:, None]
        test_x = X_t[:, pre_id].flatten()[:, None]
        test_y = Y_t[:, pre_id].flatten()[:, None]
        print(test_t[0])

        g_p, g_u, g_v = gan.predict(test_x, test_y, test_t)
        np.save('../results/val/nonw/pressure{0}.npy'.format(test_t[0]), g_p)
        np.save('../results/val/nonw/velocity_u{0}.npy'.format(test_t[0]), g_u)
        np.save('../results/val/nonw/velocity_v{0}.npy'.format(test_t[0]), g_v)