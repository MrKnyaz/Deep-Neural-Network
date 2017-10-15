# README #

Deep neural network, supports any number of hidden layers including 0 for linear model.
Has different types of output: binary, softmax and linear for regression problems.
Supports minibatches and different optimizers like "gradient", "gradient with momentum" and "adam"

### How to use ###
Model accepts data using NxM shape, where N is the number of features and M is the number of examples.  
Example:  
  
`layers_dims = [train_X.shape[0], 60, 1]  #network with 1 hidden layer containing 60 items. For linear model dims would be [train_X.shape[0], 1]`  
`deep_nn = DeepNN(layers_dims, batch={"size":256, "randomize":False}, activation_hidden="relu", activation_out="sigmoid", lambd=0.001, optimizer="adam")`  
`costs = deep_nn.train(train_X, train_Y, num_iterations=1000, learning_rate=0.007, print_cost={"print":True, "period":100})`  
`predictions = deep_nn.predict(train_X)`  
`plt.plot(costs)`


DeepNNTF - is a tensorflow based model
Model accepts data using MxN shape, where N is the number of features and M is the number of examples.
Example:

`layers_dims = [train_X.shape[1], 60, 1]  #network with 1 hidden layer containing 60 items. For linear model dims would be [train_X.shape[1], 1]`  
`deep_nn = DeepNNTF(layers_dims, batch={"size":256, "randomize":False}, activation_hidden="relu", activation_out="softmax", lambd=0.001, optimizer="adam")`  
`costs = deep_nn.train(train_X, train_Y, num_iterations=1000, learning_rate=0.007, print_cost={"print":True, "period":100})`  
`predictions = deep_nn.predict(train_X)`  
`plt.plot(costs)`  

You can set activation_out parameter as "linear" if you need regression model.

