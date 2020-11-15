import numpy as np

import matplotlib.pyplot as plt


def sigmoid(x, beta):
    #return 1.0 / (1 + np.exp(-2 * beta * x))
     return np.tanh(beta*x)
    # return np.maximum(x, 0)


def sigmoid_derivative(x, beta):
    #return 2 * beta * x * (1 - x)
    return beta*(1 - np.power(x, 2))


# x[x <= 0] = 0
# x[x > 0] = 1
# return x


def print_letter(pattern):
    for i in range(7):
        s = ""
        for j in range(8):
            val = pattern[i * 8 + j]
            s += "# " if val >= 0.01 else ". "
        print(s)


class Network:

    # -n_of_inputs: dimension de un input
    # -hidden_layers: array indicando cuantas neurona tiene cada layer
    #                 ej: [3, 2] son 2 capaz con 3 y 2 neuronas respectivamente
    # -n_of_outputs: neuronas de la ultima capa
    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs, betas):
        self.n_of_inputs = n_of_inputs
        self.hidden_layers = hidden_layers
        self.n_of_outputs = n_of_outputs
        self.betas = betas
        # Array con la cantidad de neuronas de cada capa
        neurons_of_layer = [n_of_inputs] + hidden_layers + [n_of_outputs]

        weights = []
        biases = []
        last_deltas = []
        derivatives = []
        last_derivatives = []
        deltas = []
        for i in range(len(neurons_of_layer) - 1):
            w = np.random.rand(neurons_of_layer[i], neurons_of_layer[i + 1])
            b = np.random.rand(neurons_of_layer[i + 1], 1)
            d = np.zeros((neurons_of_layer[i], neurons_of_layer[i + 1]))
            deltas_i = np.zeros((neurons_of_layer[i + 1], 1))
            deltas.append(deltas_i)
            last_deltas.append(deltas_i)
            derivatives.append(d)
            last_derivatives.append(d)
            weights.append(w)
            biases.append(b)

        self.weights = weights
        self.biases = biases
        self.last_biases = biases
        self.derivatives = derivatives
        self.last_derivatives = last_derivatives
        self.deltas = deltas
        self.last_deltas = last_deltas

        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        self.activations = activations

        print("weights: {}".format(weights))
        # print("biases: {}".format(biases))
        # print("derivatives: {}".format(derivatives))
        # print("deltas: {}".format(deltas))
        # print("activations: {}".format(activations))

    # @jit(target="cuda")
    def train(self, inputs, outputs, epochs, eta, adaptive_lr=False):

        loss = []
        x = []
        etas = []
        #error = np.zeros((1, 56))

        neg_error = 0
        pos_error = 0
        for i in range(epochs):
            total_error = 0
            for j, input_ in enumerate(inputs):
                # print("input:")
                # print_letter(input_)
                predicted_output = self.predict(input_)
                error = outputs[j] - predicted_output

                self.back_propagate(error)

                self.update_weights(eta)
                error = self.mean_square_error(outputs[j], predicted_output)
                #print("Error:",error)
                total_error += error

            print("Error: {} at epoch {}".format(total_error / len(inputs), i + 1))

            if adaptive_lr:
                etas.append(eta)
                eta = self.exp_decay(i, etas[0])
            loss.append(total_error / len(inputs))
            x.append(i)
        plt.plot(x, loss)
        plt.show()

    def predict(self, input_):

        self.activations[0] = input_

        for i, w in enumerate(self.weights):
            x = np.dot(w.T, input_) + self.biases[i].T
            # print("x",x)
            # x = x / np.linalg.norm(x)
            x = x.reshape(x.shape[1])
            input_ = sigmoid(x, self.betas[i])

            self.activations[i + 1] = input_
        # print("predicted: ",self.activations[-1])
       # print("output:",self.activations[-1])
        return self.activations[-1]

    # @jit(target="cuda")
    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            output = self.activations[i + 1]

            delta = sigmoid_derivative(output, self.betas[i]) * error  # g'(h)*error
            self.last_deltas[i] = np.copy(self.deltas[i]).reshape(delta.shape[0], -1).T

            self.deltas[i] = delta.reshape(delta.shape[0], -1).T

            inputs = self.activations[i]
            inputs = inputs.reshape(inputs.shape[0], -1)
            self.last_derivatives[i] = self.derivatives[i]
            self.derivatives[i] = np.dot(inputs, self.deltas[i])

            error = np.dot(self.deltas[i], self.weights[i].T)
            error = error / np.linalg.norm(error)
            error = error.reshape(error.shape[1])

    # @jit(target="cuda")
    def update_weights(self, eta):

        for i in range(len(self.weights)):
            delta_w = eta * self.derivatives[i] 
            self.weights[i] += delta_w
            self.biases[i] += eta*self.deltas[i].reshape(self.biases[i].shape)

    # @jit(target="cuda")
    def mean_square_error(self, expected, predicted_output):
        return np.average(np.power(expected - predicted_output,2) )

    # @jit(target="cuda")
    def exp_decay(self, epoch, eta):
        k = 0.001
        x = np.exp(-k * epoch)
        return eta * x

    def reset(self, n_of_inputs, hidden_layers, n_of_outputs):
        self.__init__(n_of_inputs, hidden_layers, n_of_outputs)

