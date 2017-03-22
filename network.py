"""
This is designed to implement the stochastic gradient descent learning
algorithm for neural network.  Gradients are calculated using backpropagation. 
"""

import random
import numpy as np

class Network(object):
    ## this is the network class
    #constructor
    def __init__(self, sizes):
    
        """The list sizes contains the number of neurons in the
        respective layers of the network. The biases and weights for the
        network are initialized randomly."""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #all layers but first one need bias
        self.weights = [np.random.randn(y, x)  for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learn_rate, test_data=None):
        """Training the neural network using stochastic  gradient descent. """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            #making batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            
            #running mini batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate)
            
            #testing only if test data given
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch,learn_rate):
        """Update the network's weights and biases by applying gradient descent using backpropagation algorithim"""
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]
        
        
        for x, y in mini_batch:
            delta_n_b, delta_n_w = self.backprop(x, y)
             
            n_b = [nb+dnb for nb, dnb in zip(n_b, delta_n_b)]
            
            
            n_w = [nw+dnw for nw, dnw in zip(n_w, delta_n_w)]
           
        self.weights = [w-(learn_rate/len(mini_batch))*nw for w, nw in zip(self.weights, n_w)]
        
        
        self.biases = [b-(learn_rate/len(mini_batch))*nb for b, nb in zip(self.biases, n_b)]

    def backprop(self, x, y):
        """Return a tuple (n_b, n_w) representing the  gradient for the cost function"""
        
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer z = W*x + b
        """ z and activations are stored """
        for b, w in zip(self.biases, self.weights):
            #calculating z
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
            
        # backward pass
        
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        n_b[-1] = delta
        n_w[-1] = np.dot(delta, activations[-2].transpose())
        
        #updating
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            n_b[-l] = delta
            n_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (n_b, n_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        
        #storing results
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        #print(test_results)
        
        #checking how many are correct
        a = [int(x == y) for (x, y) in test_results]

        return sum(a)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
        
        
        
        
        
        
        

''' we can add diff activation functions here  in place of sigmoid and use them'''


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

"""Derivative of the sigmoid function."""
def sigmoid_prime(z):
     return sigmoid(z)*(1-sigmoid(z))
