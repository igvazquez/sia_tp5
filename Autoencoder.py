from MLP import Network
import numpy as np


def sigmoid(x):
    # return 1.0 / (1 + np.exp(-x))
    return np.tanh(x)
    # return np.maximum(0, x)


class Autoencoder:

    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs, betas, max_error):
        rev = hidden_layers
        rev = rev[::-1]
        hidden_layers.append(n_of_outputs)
        hidden_layers.extend(rev)

        self.autoencoder = Network(n_of_inputs, hidden_layers, n_of_inputs, max_error)
        self.n_of_inputs = n_of_inputs
        self.n_of_outputs = n_of_outputs
        self.hl = hidden_layers
        self.neurons_of_layer = [self.n_of_inputs] + self.hl + [self.n_of_inputs]
        print("neurons_of_layer:", self.neurons_of_layer)

    def train(self, inputs, outputs, epochs, eta, K, a, b, adaptive_lr=False):
        self.autoencoder.train(inputs, outputs, epochs, eta, K, a, b, adaptive_lr)

    def decode(self, input_):
        # return self.autoencoder.predict(input_)
        weights = self.autoencoder.weights
        weights = weights[int(len(weights) / 2):]

        b = self.autoencoder.biases
        b = b[int(len(b) / 2):]

        neurons_of_layer = self.neurons_of_layer[int(len(self.neurons_of_layer) / 2):]

        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        activations[0] = input_

        for i, w in enumerate(weights):
            x = np.dot(w.T, input_) + b[i].T
            x = x.reshape(x.shape[1])
            input_ = sigmoid(x)
            activations[i + 1] = input_

        return activations[-1]

    def encode(self, input_):
        weights = self.autoencoder.weights
        weights = weights[:int(len(weights) / 2)]

        b = self.autoencoder.biases
        b = b[:int(len(b) / 2)]

        neurons_of_layer = self.neurons_of_layer[:int(len(self.neurons_of_layer) / 2) + 1]

        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        activations[0] = input_

        for i, w in enumerate(weights):
            x = np.dot(w.T, input_) + b[i].T
            x = x.reshape(x.shape[1])
            input_ = sigmoid(x)
            activations[i + 1] = input_

        return activations[-1]
