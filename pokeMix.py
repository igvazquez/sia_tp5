import os
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data

def show_image(image):
    plt.imshow(np.clip(image+0.5,0,1))
    plt.show()

data_directory = 'poke/mix/'

filenames = [os.path.join(data_directory, file_i)
             for file_i in os.listdir(data_directory)
             if '.jpg' in file_i]

imgs = [plt.imread(f) for f in filenames]

Xs = np.array(imgs)

plt.figure(figsize=(10, 10))
for i in range(len(imgs)):
    plt.imshow(imgs[i])
    plt.show()

test = []
for i in range(len(Xs)):
    test.append(((Xs[i].astype('float32') / 255.0)-0.5).reshape(12288))

# test = (((Xs[0] - np.mean(Xs[0])) / np.std(Xs[0])).reshape(-1, 12288))
# plt.imshow(test[0].reshape(64, 64, 3))
# plt.show()
hidden_layer = [192]

ae = Autoencoder(64 * 64 * 3, hidden_layer, 64, 0.00000002)

ae.train(test, test, 50, 0.0005,5, 0.5,10, 0.1, True)

outputs = []
latent_space = []
for inp in range(len(test)):
    encoded_input = ae.encode(test[inp])
    latent_space.append(encoded_input)
    outputs.append(ae.decode(encoded_input))
    show_image(outputs[inp].reshape(64,64,3))

bulb = latent_space[0]
caterpie = latent_space[1]

step = np.subtract(bulb, caterpie) / 9

output = []
for i in range(10):
    # Diagonal
    out = ae.decode(np.subtract(bulb, step*i)).reshape(64, 64, 3)
    show_image(out.reshape(64,64,3))