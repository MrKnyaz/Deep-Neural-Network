# README #

Deep neural network, supports any number of hidden layers including 0 for linear model.
Has different types of output: binary, softmax and linear for regression problems.
Supports minibatches and different optimizers like "gradient", "gradient with momentum" and "adam"

### How to use ###
Model uses data using NxM shape where N - is number of features and M - number of examples.  
Example:  
'''
layers_dims = [train_X.shape[0], 60, 1]
deep_nn = DeepNN()
costs = deep_nn.train(train_X, train_Y, layers_dims, num_iterations=2000, learning_rate=0.007, batch={"size":2048, "randomize":False}, print_cost={"print":True, "period":100}, activation_hidden="leaky_relu", activation_out="sigmoid", lambd=0.001, optimizer="adam")
predictions = deep_nn.predict(train_X)
plt.plot(costs)
'''  
You can set activation_out parameter as "linear" if you need regression model

