import os
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data


def montage(images, saveto='montage.png'):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


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

#
# for i in range(2):
#     plt.subplot(3, 3, (2 + i))
#     plt.pcolor(ae.decode([bulb[0]-step[0]*(i+1), bulb[1]]).reshape(7, 5), cmap='gray')
#     plt.subplot(3, 3, (4 + i * 3))
#     plt.pcolor(ae.decode([bulb[0], bulb[1]-step[1]*(i+1)]).reshape(7, 5), cmap='gray')
#     plt.subplot(3, 3, (6 + i * 2))
#     plt.pcolor(ae.decode([bulb[0]-step[0]*(2-i), bulb[1]-step[1]*(1+i)]).reshape(7, 5), cmap='gray')
#
# plt.tight_layout()
# plt.show()

# print(Xs.shape)
#
# # load the image
# img = Image.open('pokemon/bulbasaur.png')
# # convert image to numpy array
# data = np.asarray(img)
#
# inputs = []
# inputs.append(np.asarray(data).reshape(-1))
# inputs[0] = inputs[0] / np.linalg.norm(inputs[0])
#
# hidden_layer = [2048, 1024, 512]
# betas = np.random.random_sample((1, 2 * len(hidden_layer) + 3)) / 100
# ae = Autoencoder(64 * 64, [2048, 1024, 512], 256, betas[0], 0.02)
#
# ae.train(np.asarray(inputs), np.asarray(inputs), 100, 0.00045, 10, 0.5, 0.1, True)
#
# outputs = []
# for inp in range(len(inputs)):
#     encoded_input = ae.encode(inputs[inp])
#     outputs.append(ae.decode(encoded_input))
#
# # create Pillow image
# image2 = Image.fromarray(outputs[0].reshape(64, 64))
#
# # display the array of pixels as an image
# pyplot.imshow(image2)
# pyplot.show()
