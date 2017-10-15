import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt


class DeepNNTF(object):

    def __init__(self, layers, activation_hidden="relu", activation_out="sigmoid", lambd=0.01, optimizer="gradient", beta1=0.9, beta2=0.999, batch={"size": 0, "randomize": False}):
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.randomize_batch = batch["randomize"]
        self.batch_size = batch["size"]
        self.optimizer_name = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd = lambd
        self.layers = layers
        self.L = len(layers) - 1    # number of layers
        self.t = 0                  # adam iteration coefficient
        self.parameters = None
        self.trained_params = None
        self.cache = None           # cache for A0 Z1 A1 Z2 A2 ...
        self.X = None
        self.Y = None
        self.cost = None
        self.optimizer = None
        self.learning_rate = None
        self.build_graph()

    def build_graph(self):
        ops.reset_default_graph()
        self.parameters = self.initialize_parameters()
        self.X, self.Y, self.learning_rate = self.create_placeholders()
        self.cache = self.forward_propagation()
        self.cost = self.build_cost()
        self.cost += self.l2_regularization()
        self.optimizer = self.choose_optimizer()

    def initialize_parameters(self):
        parameters = {}
        for l in range(1, self.L + 1):
            parameters["W" + str(l)] = tf.get_variable("W" + str(l), [self.layers[l - 1], self.layers[l]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["b" + str(l)] = tf.get_variable("b" + str(l), [1, self.layers[l]], initializer=tf.zeros_initializer())
        return parameters

    def assign_trained_params(self):
        assignments = []
        for l in range(1, self.L + 1):
            assignments.append(tf.assign(self.parameters["W" + str(l)], self.trained_params["W" + str(l)]))
            assignments.append(tf.assign(self.parameters["b" + str(l)], self.trained_params["b" + str(l)]))
        return assignments

    def create_placeholders(self):
        X = tf.placeholder(tf.float32, shape=[None, self.layers[0]], name="X")
        Y = tf.placeholder(tf.float32, shape=[None, self.layers[-1]], name="Y")
        learning_rate = tf.placeholder(tf.float32, shape=[])
        return X, Y, learning_rate

    def forward_propagation(self):
        cache = {"A0": self.X}
        for l in range(1, self.L + 1):
            cache["Z" + str(l)] = tf.matmul(cache["A" + str(l - 1)], self.parameters["W" + str(l)]) + self.parameters["b" + str(l)]
            if l < self.L:
                cache["A" + str(l)] = self.g(cache["Z" + str(l)], self.activation_hidden)
            else:
                cache["A" + str(l)] = self.g(cache["Z" + str(l)], self.activation_out)
        return cache

    def g(self, Z,  activation):
        if activation == "relu":
            return tf.nn.relu(Z)
        elif activation == "tanh":
            return tf.nn.tanh(Z)
        elif activation == "sigmoid":
            return tf.nn.sigmoid(Z)
        elif activation == "softmax":
            return tf.nn.softmax(Z)
        else:
            return Z

    def build_cost(self):
        Z = self.cache["Z" + str(self.L)]
        # m = tf.shape(Z)[0]
        if self.activation_out == "sigmoid":
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=self.Y))
        elif self.activation_out == "softmax":
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=self.Y))
        else:  # linear
            return tf.reduce_mean(tf.pow(Z - self.Y, 2)) / 2

    def l2_regularization(self):
        l2_cost = 0
        for l in range(1, self.L + 1):
            l2_cost += tf.nn.l2_loss(self.parameters["W" + str(l)])
        return self.lambd * l2_cost

    def choose_optimizer(self):
        if self.optimizer_name=="adam":
            optimizer = tf.train.AdamOptimizer(beta1=self.beta1, beta2=self.beta2, learning_rate=self.learning_rate).minimize(self.cost)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        return optimizer

    def train(self, X, Y, num_iterations=1000, learning_rate=0.01, print_cost={"print": True, "period": 1000}, stop_treshold=0.0):
        costs = []
        print_cost_period = print_cost["period"]
        print_cost = print_cost["print"]
        m = X.shape[0]
        if self.batch_size == 0:
            batches = [(X, Y)]
            num_of_minibatches = 1
        else:
            seed = 11
            batches = self.generate_mini_batches(X, Y, self.batch_size, seed)
            num_of_minibatches = int(m / self.batch_size)

        prev_cost = 0
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            if self.trained_params:  # For multiple calls of "train" method, trained_parameters will be reused(this is for jupyter notebooks mainly)
                sess.run(self.assign_trained_params())

            for i in range(num_iterations):
                cost = 0
                for batch in batches:
                    batch_X, batch_Y = batch
                    # m_batch = batch_X.shape[0]
                    batch_cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: batch_X, self.Y: batch_Y, self.learning_rate: learning_rate})
                    cost += batch_cost / num_of_minibatches

                if self.batch_size != 0 and self.randomize_batch:
                    seed = seed + 1
                    batches = self.generate_mini_batches(X, Y, self.batch_size, seed)
                if i % print_cost_period == 0 and print_cost:
                    print("Cost after iteration %s is: %s" % (i, cost))
                if i % (print_cost_period / 10) == 0:
                    costs.append(cost)
                if abs(prev_cost - cost) <= stop_treshold:
                    print("Final cost is %s" % cost)
                    print("Cost did not improve more than %s" % stop_treshold)
                    break
                prev_cost = cost
            self.trained_params = sess.run(self.parameters)
        return costs

    def predict(self, X):
        with tf.Session() as sess:
            sess.run(self.assign_trained_params())
            result = sess.run(self.cache["A" + str(self.L)], feed_dict={self.X: X})
        if self.activation_out == "sigmoid":
            result = result > 0.5
        return result

    def generate_mini_batches(self, X, Y, batch_size, seed=0):
        m = X.shape[0]
        batches = []
        # np.random.seed(seed)

        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]
        num_of_minibatches = math.floor(m / batch_size)

        for i in range(num_of_minibatches):
            batch_X = shuffled_X[i * batch_size:(i + 1) * batch_size, :]
            batch_Y = shuffled_Y[i * batch_size:(i + 1) * batch_size, :]
            batch = (batch_X, batch_Y)
            batches.append(batch)
        if m % batch_size != 0:
            batch_X = shuffled_X[num_of_minibatches * batch_size:, :]
            batch_Y = shuffled_Y[num_of_minibatches * batch_size:, :]
            batch = (batch_X, batch_Y)
            batches.append(batch)
        return batches


def check_parameters():
    dnn = DeepNNTF([2, 3, 1])
    print(dnn.parameters)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(dnn.parameters))
        # Good news everyone! sess.run returns values in the same form as they were passed
        print(sess.run({"PARAM1": dnn.parameters["W1"], "PARAM2": dnn.parameters["W2"]}))

def check_nn():
    dnn = DeepNNTF([2, 3, 1])
    X_train = np.arange(0, 20).reshape((10, 2))
    Y_train = np.array([x < 5 for x in range(10)]).reshape(-1, 1)
    costs = dnn.train(X_train, Y_train, 100, learning_rate=0.01, print_cost={"print": True, "period": 10})
    plt.plot(costs)
    costs = dnn.train(X_train, Y_train, 100, learning_rate=0.01, print_cost={"print": True, "period": 10})
    plt.plot(costs)
    plt.show()
    # print(X_train)
    # print(Y_train)

#check_parameters()
#check_nn()