import os
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from celluloid import Camera
def show_image(image):
    plt.imshow(np.clip(image+0.5,0,1))


data_directory = 'poke/evolution/'
output_directory = 'output/'

filenames = [os.path.join(data_directory, file_i)
             for file_i in os.listdir(data_directory)
             if '.jpg' in file_i]

imgs = [plt.imread(f) for f in filenames]

Xs = np.array(imgs)

# plt.figure(figsize=(10, 10))
# for i in range(len(imgs)):
#     plt.imshow(imgs[i])
#     plt.show()

test = []
for i in range(len(Xs)):
    test.append(((Xs[i].astype('float32') / 255.0)-0.5).reshape(12288))
print("test size",len(test))
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
    #show_image(outputs[inp].reshape(64,64,3))


fig,axs = plt.subplots()
camera = Camera(fig)
for i in range(len(latent_space)-1):
    l1 = latent_space[i]
    l2 = latent_space[i+1]
    step = np.subtract(l1, l2) / 8
    for j in range(9):
        out = ae.decode(np.subtract(l1, step*j)).reshape(64, 64, 3)
        plt.imshow(np.clip(out.reshape(64,64,3) + 0.5, 0, 1))
        camera.snap()
animation = camera.animate(interval=10, repeat=True)
plt.show()
