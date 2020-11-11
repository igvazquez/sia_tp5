from MLP import Network
class Autoencoder:

    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs):
        hidden_layers.append(n_of_outputs)
        hidden_layers.extend(hidden_layers[1::-1])
        print(hidden_layers)
        self.autoencoder = Network(n_of_inputs, hidden_layers, n_of_inputs)

    def train(self, inputs, outputs, epochs, eta, adaptive_lr=False):
        self.autoencoder.train(inputs, outputs, epochs, eta, adaptive_lr)

    def predict(self, input_):
        return self.autoencoder.predict(input_)
