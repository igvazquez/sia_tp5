import numpy as np
import pandas as pd
from Autoencoder import Autoencoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('fonttesting.txt', delimiter="\n", header=None, dtype=str)

df = np.array(df)
data = df.reshape(32, 7)
print("df", data)
input_numbers = []
for i in range(len(data)):
    input_numbers.append("".join(np.squeeze(np.asarray(data[i]))))

for i in range(len(input_numbers)):
    input_numbers[i] = list(input_numbers[i])
    input_numbers[i] = [-1 if j == '0' else int(j) for j in input_numbers[i]]
    norm = np.linalg.norm(input_numbers[i])
    # if norm > 0:
    #     input_numbers[i] = input_numbers[i] / np.linalg.norm(input_numbers[i])

output_numbers = input_numbers
hidden_layer = [35,20, 10, 5]
# 7*5 pixeles
ae = Autoencoder(35, hidden_layer, 2, 0.02)
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 15000, 0.0005, 10, 0, 10,0, False)

outputs = []
latent_space = []
labels = ["At", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
          "V", "W", "X", "Y", "Z", "[", "\\", "^", "_", "Space"]
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(input_numbers[inp])
    latent_space.append(encoded_input)
    outputs.append(ae.decode(encoded_input))

# Grafico del espacio latente en 2 dimensiones
x = []
y = []
for i in range(len(latent_space)):
    x.append(latent_space[i][0])
    y.append(latent_space[i][1])

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]))

# Grafico acercando D a B
b = latent_space[1]
d = latent_space[13]

step = np.subtract(b, d) / 2

plt.figure(figsize=(14, 14))
for i in range(3):
    # Diagonal
    plt.subplot(3, 3, (1+4*i))
    plt.pcolor(ae.decode(np.subtract(b, step*i)).reshape(7, 5), cmap='binary')

for i in range(2):
    plt.subplot(3, 3, (2 + i))
    plt.pcolor(ae.decode([b[0]-step[0]*(i+1), b[1]]).reshape(7, 5), cmap='binary')
    plt.subplot(3, 3, (4 + i * 3))
    plt.pcolor(ae.decode([b[0], b[1]-step[1]*(i+1)]).reshape(7, 5), cmap='binary')
    plt.subplot(3, 3, (6 + i * 2))
    plt.pcolor(ae.decode([b[0]-step[0]*(2-i), b[1]-step[1]*(1+i)]).reshape(7, 5), cmap='binary')

plt.tight_layout()
plt.show()

# Grafico de las letras
for i, out in enumerate(outputs):
    outputs[i] = np.array(out).reshape((7, 5))

n_letters = len(input_numbers)

for i, input_ in enumerate(input_numbers):
    input_numbers[i] = np.array(input_).reshape((7, 5))

for i in range(32):
    fig, ax = plt.subplots(ncols=2)
    sns.heatmap(input_numbers[i], cbar=False, cmap='binary', ax=ax[0])
    sns.heatmap(outputs[i], cbar=True, cmap='binary', ax=ax[1])
    plt.show()
