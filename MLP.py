import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    # return 1.0 / (1 + np.exp(-x))
    return np.tanh(x)
    # return np.maximum(0, x)

def sigmoid_derivative(x):
    # return x * (1 - x)
    return 1 - np.power(x, 2)
    # return 1.0*(x>0)

class Network:

    # -n_of_inputs: dimension de un input
    # -hidden_layers: array indicando cuantas neurona tiene cada layer
    #                 ej: [3, 2] son 2 capaz con 3 y 2 neuronas respectivamente
    # -n_of_outputs: neuronas de la ultima capa
    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs):
        self.n_of_inputs = n_of_inputs
        self.hidden_layers = hidden_layers
        self.n_of_outputs = n_of_outputs

        # Array con la cantidad de neuronas de cada capa
        neurons_of_layer = [n_of_inputs] + hidden_layers + [n_of_outputs]

        weights = []
        biases = []
        derivatives = []
        deltas = []
        for i in range(len(neurons_of_layer) - 1):
            w = np.random.normal(loc=0.0, scale=np.sqrt(2 / (neurons_of_layer[i] + neurons_of_layer[i + 1])),
                                 size=(neurons_of_layer[i], neurons_of_layer[i + 1]))
            # w = np.random.rand(neurons_of_layer[i], neurons_of_layer[i + 1])
            b = np.random.rand(neurons_of_layer[i + 1], 1)
            d = np.zeros((neurons_of_layer[i], neurons_of_layer[i + 1]))
            deltas_i = np.zeros((neurons_of_layer[i + 1], 1))
            deltas.append(deltas_i)
            derivatives.append(d)
            weights.append(w)
            biases.append(b)
        self.weights = weights
        self.biases = biases
        self.derivatives = derivatives
        self.deltas = deltas

        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        self.activations = activations

        print("weights: {}".format(weights))
        print("biases: {}".format(biases))
        print("derivatives: {}".format(derivatives))
        print("deltas: {}".format(deltas))
        print("activations: {}".format(activations))

    def predict(self, input_):

        self.activations[0] = input_

        for i, w in enumerate(self.weights):
            x = np.dot(input_, w) + self.biases[i].T
            x = x.reshape(x.shape[1])
            input_ = sigmoid(x)
            self.activations[i + 1] = input_

        return self.activations[-1]

    def train(self, inputs, outputs, epochs, eta, adaptive_lr=False):

        loss = []
        x = []
        etas = []
        for i in range(epochs):
            total_error = 0
            for j, input_ in enumerate(inputs):
                predicted_output = self.predict(input_)

                error = outputs[j] - predicted_output

                self.back_propagate(error)

                self.update_weights(eta)

                total_error += self.mean_square_error(outputs[j], predicted_output)
            print("Error: {} at epoch {}".format(total_error / len(inputs), i + 1))
            if adaptive_lr:
                etas.append(eta)
                eta = self.exp_decay(i, etas[0])
            loss.append(total_error / len(inputs))
            x.append(i)
        plt.plot(x, loss)
        plt.show()

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            output = self.activations[i + 1]

            delta = sigmoid_derivative(output) * error
            self.deltas[i] = delta.reshape(delta.shape[0], -1).T

            inputs = self.activations[i]
            inputs = inputs.reshape(inputs.shape[0], -1)

            self.derivatives[i] = np.dot(inputs, self.deltas[i])

            error = np.dot(self.deltas[i], self.weights[i].T)
            error = error.reshape(error.shape[1])

    def update_weights(self, eta):

        for i in range(len(self.weights)):
            self.weights[i] += eta * self.derivatives[i]
            self.biases[i] += eta * self.deltas[i].reshape(self.biases[i].shape)

    def mean_square_error(self, expected, predicted_output):
        return np.average((expected - predicted_output) ** 2)

    def exp_decay(self, epoch, eta):
        k = 0.001
        x = np.exp(-k * epoch)
        return eta * x

    def reset(self, n_of_inputs, hidden_layers, n_of_outputs):
        self.__init__(n_of_inputs, hidden_layers, n_of_outputs)
