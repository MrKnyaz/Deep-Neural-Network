import numpy as np
import math

class DeepNN(object):

    def __init__(self):
        self.parameters = None
        self.activation_hidden = None
        self.activation_out = None

    def initialize_params(self, layers):
        # np.random.seed(3)
        parameters = {}
        for l in range(len(layers) - 1):
            parameters["W" + str(l+1)] = np.random.randn(layers[l + 1], layers[l]) * (np.sqrt(2 / layers[l]))
            parameters["b" + str(l+1)] = np.zeros((layers[l + 1], 1))
        self.parameters = parameters
        return parameters

    def initialize_velocity(self):
        L = len(self.parameters) // 2
        v = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
        return v

    def initialize_adam(self):
        L = len(self.parameters) // 2
        v = {}
        s = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
        return v, s

    def train(self, X, Y, layers, num_iterations=9000, activation_hidden="relu", activation_out="sigmoid", learning_rate=0.01, lambd=0.01, optimizer="gradient", beta1=0.9, beta2=0.999, batch={"size": 0, "randomize": False}, print_cost={"print": True, "period": 1000}):
        self.initialize_params(layers)
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        L = len(self.parameters) // 2
        t = 0
        costs = []
        randomize_batch = batch["randomize"]
        batch_size = batch["size"]
        print_cost_period = print_cost["period"]
        print_cost = print_cost["print"]
        if optimizer == "gradient_momentum":
            v = self.initialize_velocity()
        elif optimizer == "adam":
            v, s = self.initialize_adam()
        if batch_size == 0:
            batches = [(X, Y)]
        else:
            seed = 11
            batches = self.generate_mini_batches(X, Y, batch_size, seed)
        for i in range(num_iterations):
            for batch in batches:
                batch_X, batch_Y = batch
                m = batch_X.shape[1]
                cache = self.forward_propagation(batch_X)
                if activation_out != "linear":
                    cost = self.cost(cache["A" + str(L)], batch_Y)
                else:
                    cost = self.cost(cache["Z" + str(L)], batch_Y)
                cost = cost + self.l2_regularization_cost(lambd, m)
                grads = self.backward_propagation(batch_Y, cache, lambd)

                if optimizer == "gradient":
                    self.update_parameters(grads, learning_rate)
                elif optimizer == "gradient_momentum":
                    self.update_parameters_momentum(grads, learning_rate, v, beta1)
                elif optimizer == "adam":
                    t = t + 1
                    self.update_parameters_adam(grads, learning_rate, v, s, t, beta1, beta2)

            if batch_size != 0 and randomize_batch:
                seed = seed + 1
                batches = self.generate_mini_batches(X, Y, batch_size, seed)
            if i % print_cost_period == 0 and print_cost:
                print("Cost after iteration %s is: %s" % (i, cost))
            if i % (print_cost_period / 10) == 0:
                costs.append(cost)

        return costs

    def cost(self, AL, Y):
        m = Y.shape[1]
        if self.activation_out == "sigmoid":
            cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        elif self.activation_out == "softmax":
            cost = np.sum(-Y * np.log(AL)) / m
        else:  # linear
            cost = np.sum(np.power(AL - Y, 2)) / (2 * m)
        return np.squeeze(cost)

    def l2_regularization_cost(self, lambd, m):
        L = len(self.parameters) // 2
        sum = 0
        for l in range(1, L+1):
            sum = sum + np.sum(np.square(self.parameters["W" + str(l)]))
        l2_cost = sum * (lambd / (2 * m))
        return l2_cost

    def predict(self, X):
        L = len(self.parameters) // 2
        cache = self.forward_propagation(X)
        A = cache["A" + str(L)]
        if self.activation_out != "linear":
            A = A > 0.5
        return A

    def forward_propagation(self, X):
        m = X.shape[1]
        L = len(self.parameters) // 2
        A = X
        cache = {"A0": A}
        for l in range(1, L + 1):
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A) + b
            if l < L:
                A = self.g(Z, self.activation_hidden)
            else:
                A = self.g(Z, self.activation_out)
            cache["Z" + str(l)] = Z
            cache["A" + str(l)] = A
        return cache

    def backward_propagation(self, Y, cache, lambd):
        m = Y.shape[1]
        L = len(self.parameters) // 2
        AL = cache["A" + str(L)]
        dZ = AL - Y
        grads = {}
        dWL, dbL = self.parameters_derivatives(dZ, cache, m, L, lambd)
        grads["dW" + str(L)] = dWL
        grads["db" + str(L)] = dbL
        for l in reversed(range(1, L)):  # if 3 layers it will go from 2 to 1, 3rd layer we already calculated
            # calculate next dZ
            W = self.parameters["W" + str(l+1)]  # W2
            Z = cache["Z" + str(l)]  # Z1
            dA = np.dot(W.T, dZ)  # dA1 dZ2
            if self.activation_hidden == "relu":
                dZ = self.relu_derivative(dA, Z)
            elif self.activation_hidden == "leaky_relu":
                dZ = self.leaky_relu_derivative(dA, Z)
            elif self.activation_hidden == "sigmoid":
                dZ = self.sigmoid_derivative(dA, Z)
            else:
                dZ = self.tanh_derivative(dA, Z)
            dW, db = self.parameters_derivatives(dZ, cache, m, l, lambd)
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db
        return grads

    def parameters_derivatives(self, dZ, cache,  m, current_layer, lambd):
        dW = np.dot(dZ, cache["A" + str(current_layer - 1)].T) / m + (lambd / m) * self.parameters["W" + str(current_layer)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        return dW, db

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    def update_parameters_momentum(self, grads, learning_rate, v, beta1):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            v["dW" + str(l)] = v["dW" + str(l)] * beta1 + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = v["db" + str(l)] * beta1 + (1 - beta1) * grads["db" + str(l)]
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

    def update_parameters_adam(self, grads, learning_rate, v, s, t, beta1, beta2, epsilon = 1e-8):
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = v["dW" + str(l)] * beta1 + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = v["db" + str(l)] * beta1 + (1 - beta1) * grads["db" + str(l)]
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, 2))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, 2))
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

    def g(self, Z, activation):
        if activation == "relu":
            return self.relu(Z)
        elif activation == "leaky_relu":
            return self.leaky_relu(Z)
        elif activation == "tanh":
            return np.tanh(Z)
        elif activation == "sigmoid":
            return self.sigmoid(Z)
        else:
            return Z

    def relu(self, Z):
        return np.maximum(0, Z)

    def leaky_relu(self, Z):
        return np.maximum(0.01 * Z, Z)

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def sigmoid_derivative(self, dA, Z):
        A = self.sigmoid(Z)
        dZ = dA * (A * (1 - A))
        return dZ

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def leaky_relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = dZ[Z <= 0] * 0.01
        return dZ

    def tanh_derivative(self, dA, Z):
        A = np.tanh(Z)
        dZ = dA * (1 - np.power(A, 2))
        return dZ

    def generate_mini_batches(self, X, Y, batch_size, seed=0):
        m = X.shape[1]
        batches = []
        # np.random.seed(seed)

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        num_of_minibatches = math.floor(m / batch_size)

        for i in range(num_of_minibatches):
            batch_X = shuffled_X[:, i * batch_size:(i + 1) * batch_size]
            batch_Y = shuffled_Y[:, i * batch_size:(i + 1) * batch_size]
            batch = (batch_X, batch_Y)
            batches.append(batch)
        if m % batch_size != 0:
            batch_X = shuffled_X[:, num_of_minibatches * batch_size:]
            batch_Y = shuffled_Y[:, num_of_minibatches * batch_size:]
            batch = (batch_X, batch_Y)
            batches.append(batch)
        return batches

# test neural network components
def test_initialize_params():
    deep_nn = DeepNN()
    deep_nn.initialize_params([10, 4, 2])
    params = deep_nn.parameters
    print(params["W1"].shape)
    print(params["b1"].shape)
    print(params["W2"].shape)
    print(params["b2"].shape)
    print(params["W1"])

def test_forward_propagation():
    deep_nn = DeepNN()
    deep_nn.initialize_params([4, 3, 2, 1])
    X = np.random.randn(4, 1)
    cache = deep_nn.forward_propagation(X)
    print(cache["A0"].shape)
    print(cache["Z1"].shape)
    print(cache["A1"].shape)
    print(cache["Z2"].shape)
    print(cache["A2"].shape)
    print(cache)

def test_forward_and_back():
    deep_nn = DeepNN()
    deep_nn.initialize_params([4, 3, 1])
    X = np.random.randn(4, 1)
    Y = np.round(deep_nn.sigmoid(np.random.randn(1, 1)))
    cache = deep_nn.forward_propagation(X)
    cost = deep_nn.cost(cache["A" + str(2)], Y, "sigmoid")
    grads = deep_nn.backward_propagation(Y, cache)
    print(cache)
    print(grads)
    print("cost %s" % cost)

def test_generate_minibatches():
    deep_nn = DeepNN()
    X = np.array([[1,2,3,99], [4,5,6,88]])
    Y = np.array([[7,8,9,77]])
    batches = deep_nn.generate_mini_batches(X, Y, 2)
    for batch in batches:
        batch_X, batch_Y = batch
        print(batch_X)
        print(batch_Y)

# np.random.seed(2)
# test_initialize_params()
# test_forward_propagation()
# test_forward_and_back()
# test_generate_minibatches()
