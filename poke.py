import os
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data


data_directory = 'poke/'
output_directory = 'output/'

filenames = [os.path.join(data_directory, file_i)
             for file_i in os.listdir(data_directory)
             if '.jpg' in file_i]

imgs = [plt.imread(f) for f in filenames]

Xs = np.array(imgs)

plt.figure(figsize=(10, 10))
plt.imshow(imgs[0])
plt.show()
plt.imshow(imgs[1])
plt.show()

test = []
for i in range(len(Xs)):
    test.append(((Xs[i] - np.mean(Xs[i])) / np.std(Xs[i])).reshape(12288))

# test = (((Xs[0] - np.mean(Xs[0])) / np.std(Xs[0])).reshape(-1, 12288))
# plt.imshow(test[0].reshape(64, 64, 3))
# plt.show()
hidden_layer = [1024, 64]
betas = np.random.random_sample((1, 2 * len(hidden_layer) + 3)) / 100
ae = Autoencoder(64 * 64 * 3, hidden_layer, 4, betas[0], 0.02)

ae.train(test, test, 20, 0.001, 10, 0.5, 0.1, True)

outputs = []
latent_space = []
for inp in range(len(test)):
    encoded_input = ae.encode(test[inp])
    latent_space.append(encoded_input)
    outputs.append(ae.decode(encoded_input))
    unnorm_img = (outputs[inp] * np.std(test[inp]) + np.mean(test[inp])).reshape(64, 64, 3)
    plt.imshow(unnorm_img)
    plt.show()

bulb = latent_space[0]
caterpie = latent_space[1]

step = np.subtract(bulb, caterpie) / 3

output = []
for i in range(4):
    # Diagonal
    out = ae.decode(np.subtract(bulb, step*i)).reshape(64, 64, 3)
    unnorm_img = (out * np.std(test[0]) + np.mean(test[0])).reshape(64, 64, 3)
    plt.imshow(unnorm_img)
    plt.show()
