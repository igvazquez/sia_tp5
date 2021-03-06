import numpy as np
import pandas as pd
from Autoencoder import Autoencoder
import seaborn as sns
import matplotlib.pyplot as plt


def randomize(value):
    rand = 1.5 * np.random.rand() - 1.5

    return value + rand


df = pd.read_csv('fonttesting.txt', delimiter="\n", header=None, dtype=str)

df = np.array(df)
data = df.reshape(32, 7)
#print("df", data)
input_numbers = []
for i in range(len(data)):
    input_numbers.append("".join(np.squeeze(np.asarray(data[i]))))
for i in range(len(input_numbers)):
    input_numbers[i] = list(input_numbers[i])
    input_numbers[i] = [-1 if j == '0' else int(j) for j in input_numbers[i]]
    # norm = np.linalg.norm(input_numbers[i])
    # if norm > 0:
    #     input_numbers[i] = input_numbers[i] / np.linalg.norm(input_numbers[i])

random_prob = 0.1
random_func = np.vectorize(randomize)

output_numbers = np.copy(input_numbers)
for i in range(len(input_numbers)):
    input_numbers[i] = random_func(input_numbers[i])

# outputs = []
# for inp in range(len(input_numbers)):
#     encoded_input = ae.encode(input_numbers[inp])
#     outputs.append(ae.decode(encoded_input))


hidden_layer = []
# 7*5 pixeles
ae = Autoencoder(35, hidden_layer, 30, 0.000002)
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 10000, 0.01, 10, 5, 10,1, True)

# Plot the random inputs decodification
random_inputs_outputs = []
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(input_numbers[inp])
    random_inputs_outputs.append(ae.decode(encoded_input))

for i, input_ in enumerate(input_numbers):
    input_numbers[i] = np.array(input_).reshape((7, 5))

for i, out in enumerate(random_inputs_outputs):
    random_inputs_outputs[i] = np.array(out).reshape((7, 5))

# Plot the new inputs decodification

new_inputs = list(output_numbers)
for i in range(len(new_inputs)):
    new_inputs[i] = random_func(new_inputs[i])
    # print("new_inputs",new_inputs)
new_outputs = []
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(new_inputs[inp])
    new_outputs.append(ae.decode(encoded_input))
    # print("outputs", new_inputs)
for i, out in enumerate(new_outputs):
    new_outputs[i] = np.array(out).reshape((7, 5))

for i, input_ in enumerate(new_inputs):
    new_inputs[i] = np.array(input_).reshape((7, 5))
    # print("new_inputs", new_inputs)
for i in range(32):
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title('Random input')
    ax[0, 1].set_title('Output')
    ax[1, 0].set_title('New random input')
    ax[1, 1].set_title('New output')
    sns.heatmap(input_numbers[i], cbar=False, cmap='binary', ax=ax[0, 0])
    sns.heatmap(random_inputs_outputs[i], cbar=True, cmap='binary', ax=ax[0, 1])
    sns.heatmap(new_inputs[i], cbar=False, cmap='binary', ax=ax[1, 0])
    sns.heatmap(new_outputs[i], cbar=True, cmap='binary', ax=ax[1, 1])

plt.show()
plt.close('all')
print("Fin")
