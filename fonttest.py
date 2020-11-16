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
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 10000, 0.0005,50,0,10,0, False)

outputs = []
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(input_numbers[inp])
    outputs.append(ae.decode(encoded_input))

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
