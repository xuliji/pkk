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

import time

import sys

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


def mean_absolute_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.abs(pred - exact))
    return tf.reduce_mean(tf.abs(pred - exact))


def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))


def train_log(path, *inputs):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(inputs) + '\n')


class MLP:

    def __init__(self, x_data, y_data, t_data, pressure_data, velocity_u_data, velocity_v_data,
                 x_PDEs_data, y_PDEs_data, t_PDEs_data,
                 x_inlet_data, y_inlet_data, t_inlet_data, velocity_u_inlet_data, velocity_v_inlet_data):

        self.batch_size = 10000
        self.learning_rate = 0.001
        self.epochs = 30000

        # define layers
        self.layers = [3] + 10 * [256] + [3]
        self.layers_num = len(self.layers)

        self.model_root_path = '../models/mlp'
        self.model_path = '../models/mlp/mlp{0}.ckpt'

        self.Reynolds = 100

        # input data
        [self.x_data, self.y_data, self.t_data,
         self.pressure_data, self.velocity_u_data, self.velocity_v_data] = [x_data, y_data, t_data,
                                                                            pressure_data, velocity_u_data, velocity_v_data]

        [self.t_inlet_data, self.x_inlet_data, self.y_inlet_data,
         self.velocity_u_inlet_data, self.velocity_v_inlet_data] = [t_inlet_data,
                                                                                              x_inlet_data,
                                                                                              y_inlet_data,
                                                                                              velocity_u_inlet_data,
                                                                                              velocity_v_inlet_data]

        [self.t_PDEs_data, self.x_PDEs_data, self.y_PDEs_data] = [t_PDEs_data, x_PDEs_data, y_PDEs_data]

        # placeholder
        [self.x_tf, self.y_tf, self.t_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        [self.pressure_tf, self.velocity_u_tf, self.velocity_v_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _
                                                                      in range(3)]

        [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in
                                                               range(3)]

        [self.velocity_u_inlet_tf, self.velocity_v_inlet_tf] = [
            tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]

        [self.t_PDEs_tf, self.x_PDEs_tf, self.y_PDEs_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in
                                                            range(3)]

        # build neural network
        self.neural_network = NN(self.layers, self.layers_num)
        self.saver = tf.train.Saver()

        # MLP
        [self.pressure_,
         self.velocity_u_,
         self.velocity_v_] = self.neural_network.operator(self.x_tf,
                                                          self.y_tf,
                                                          self.t_tf)

        # boundary conditions
        [_,
         self.velocity_u_inlet_,
         self.velocity_v_inlet_] = self.neural_network.operator(self.x_inlet_tf,
                                                                self.y_inlet_data,
                                                                self.t_inlet_tf)

        # PDEs
        [self.pressure_PDEs_,
         self.velocity_u_PDEs_,
         self.velocity_v_PDEs_] = self.neural_network.operator(self.x_PDEs_tf,
                                                               self.y_PDEs_tf,
                                                               self.t_PDEs_tf)

        # PDEs residual
        [self.e_1,
         self.e_2,
         self.e_3] = Navier_Stokes(self.pressure_PDEs_,
                                   self.velocity_u_PDEs_,
                                   self.velocity_v_PDEs_,
                                   self.x_PDEs_tf,
                                   self.y_PDEs_tf,
                                   self.t_PDEs_tf,
                                   self.Reynolds)

        # define loss and optimizer
        self.loss_MLP = mean_squared_error(self.pressure_, self.pressure_tf) + \
                        mean_squared_error(self.velocity_u_, self.velocity_u_tf) + \
                        mean_squared_error(self.velocity_v_, self.velocity_v_tf)

        self.loss_BC = mean_squared_error(self.velocity_u_inlet_, self.velocity_u_inlet_tf) + \
                       mean_squared_error(self.velocity_v_inlet_, self.velocity_v_inlet_tf)

        self.loss_PDEs = mean_squared_error(self.e_1, 0.0) + \
                         mean_squared_error(self.e_2, 0.0) + \
                         mean_squared_error(self.e_3, 0.0)

        self.loss = self.loss_PDEs + self.loss_BC + self.loss_MLP
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.session = tf_session()

    def train(self):
        data_size = self.t_data.shape[0]

        running_time = 0

        log_save_path = '../results/mlp/loss.txt'

        for epoch in range(self.epochs):
            start_time = time.time()
            id_x = np.random.choice(data_size, self.batch_size)
            id_x_PDEs = np.random.choice(data_size, self.batch_size)

            (x_data_batch,
             y_data_batch,
             t_data_batch) = (self.x_data[id_x, :],
                              self.y_data[id_x, :],
                              self.t_data[id_x, :])

            (pressure_data_batch,
             velocity_u_data_batch,
             velocity_v_data_batch) = (self.pressure_data[id_x, :],
                                       self.velocity_u_data[id_x, :],
                                       self.velocity_v_data[id_x, :])

            (x_PDEs_batch,
             y_PDEs_batch,
             t_PDEs_batch) = (self.x_PDEs_data[id_x_PDEs, :],
                              self.y_PDEs_data[id_x_PDEs, :],
                              self.t_PDEs_data[id_x, :])

            tf_dict = {self.x_tf: x_data_batch,
                       self.y_tf: y_data_batch,
                       self.t_tf: t_data_batch,
                       self.pressure_tf: pressure_data_batch,
                       self.velocity_u_tf: velocity_u_data_batch,
                       self.velocity_v_tf: velocity_v_data_batch,
                       self.t_inlet_tf: self.t_inlet_data,
                       self.x_inlet_tf: self.x_inlet_data,
                       self.y_inlet_tf: self.y_inlet_data,
                       self.velocity_u_inlet_tf: self.velocity_u_inlet_data,
                       self.velocity_v_inlet_tf: self.velocity_v_inlet_data,
                       self.x_PDEs_tf: x_PDEs_batch,
                       self.y_PDEs_tf: y_PDEs_batch,
                       self.t_PDEs_tf: t_PDEs_batch}

            [_, p, v_u, v_v] = self.session.run([self.optimizer,
                                                    self.pressure_,
                                                    self.velocity_u_,
                                                    self.velocity_v_], tf_dict)

            if (epoch + 1) % 10 == 0:
                epochs_10_time = time.time() - start_time

                [loss_value, loss_pdes, loss_mlp, loss_bc] = self.session.run([self.loss, self.loss_PDEs, self.loss_MLP, self.loss_BC], tf_dict)

                train_log(log_save_path, loss_value, loss_pdes, loss_mlp, loss_bc)

                full_time = time.time() - start_time
                running_time += full_time / 3600

                print('epoch: %d, loss: %.4e, training time: %.4f s, full time: %.4f,  running time: %.4f h'
                      % (epoch + 1, loss_value, epochs_10_time, full_time, running_time))
                sys.stdout.flush()

            if (epoch + 1) % 10000 == 0:
                model_path = self.model_path.format(str(epoch + 1))
                save_path = self.saver.save(self.session, model_path)
                print('epoch: %d, model has been saved in file %s' % (epoch + 1, save_path))

    def predict(self, x_data, y_data, t_data):
        ckpt = tf.train.latest_checkpoint(self.model_root_path)
        self.saver.restore(self.session, ckpt)

        tf_dict = {self.x_tf: x_data,
                   self.y_tf: y_data,
                   self.t_tf: t_data}

        pressure_pred = self.session.run(self.pressure_, tf_dict)
        velocity_u_pred = self.session.run(self.velocity_u_, tf_dict)
        velocity_v_pred = self.session.run(self.velocity_v_, tf_dict)

        return pressure_pred, velocity_u_pred, velocity_v_pred


class NN:
    def __init__(self, layers, layers_num):
        self.layers = layers
        self.layers_num = layers_num

        self.W = []
        self.B = []
        self.G = []

        for i in range(self.layers_num - 1):
            w = tf.Variable(xavier_init([layers[i], layers[i + 1]]), dtype=tf.float32, name='weight')
            b = tf.Variable(tf.zeros(shape=[1, layers[i + 1]]), dtype=tf.float32, name='bias')
            g = tf.Variable(tf.ones(shape=[1, layers[i + 1]]), dtype=tf.float32, name='gamma')
            self.W.append(w)
            self.B.append(b)
            self.G.append(g)

    def operator(self, *inputs):
        # concat
        H = tf.concat(inputs, axis=1)

        print(self.layers_num)

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

        Output_Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        return Output_Y


def gradient(Y, x):
    dummy = tf.ones_like(Y)
    grad = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(grad, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


# N-S
# u_t + u * u_x + v * u_y = - p_x + 1/Re * (u_x_x + u_y_y)
# v_t + u * v_x + v * v_y = - p_y + 1/Re * (v_x_x + v_y_y)
# u_x + v_y = 0
def Navier_Stokes(p, u, v, x, y, t, Re):
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


def tf_session():
    # tf session
    sess = tf.Session()

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


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
    id_pde = np.random.choice(data_size, data_size, replace=False)

    t_train_data = t_train[idx]
    x_train_data = x[idx]
    y_train_data = y[idx]

    t_pde_data = t_train[id_pde]
    x_pde_data = x[id_pde]
    y_pde_data = y[id_pde]

    p_train_data = p_train[idx]
    u_train_data = u_train[idx]
    v_train_data = v_train[idx]

    t_inlet_train = t_train_data[x == x.min()][:, None]
    x_inlet_train = x_train_data[x == x.min()][:, None]
    y_inlet_train = y_train_data[x == x.min()][:, None]

    velocity_u_inlet_train = u_train_data[x == x.min()][:, None]
    velocity_v_inlet_train = v_train_data[x == x.min()][:, None]

    mlp = MLP(x_train_data, y_train_data, t_train_data, p_train_data, u_train_data, v_train_data,
               x_pde_data, y_pde_data, t_pde_data,
              x_inlet_train, y_inlet_train, t_inlet_train,
              velocity_u_inlet_train, velocity_v_inlet_train)

    # mlp.train()

    for pre_id in range(time_step_test):
        test_t = T_test_t[:, pre_id].flatten()[:, None]
        test_x = X_t[:, pre_id].flatten()[:, None]
        test_y = Y_t[:, pre_id].flatten()[:, None]
        print(test_t[0])

        g_p, g_u, g_v = mlp.predict(test_x, test_y, test_t)
        np.save('../results/val/mlp/pressure{0}.npy'.format(test_t[0]), g_p)
        np.save('../results/val/mlp/velocity_u{0}.npy'.format(test_t[0]), g_u)
        np.save('../results/val/mlp/velocity_v{0}.npy'.format(test_t[0]), g_v)