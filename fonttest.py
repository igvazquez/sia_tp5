import numpy as np
import pandas as pd
from Autoencoder import Autoencoder
import seaborn as sns
import matplotlib.pyplot as plt

betas = np.random.random_sample((1, 19))
print(betas)

df = pd.read_csv('fonttesting.txt', delimiter="\n", header=None, dtype=str)

df = np.array(df)
data = df.reshape(6, 7)
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

# 7*5 pixeles
ae = Autoencoder(35, [56, 56], 56, betas[0])
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 10000, 0.1, False)

outputs = []
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(input_numbers[inp])
    outputs.append(ae.decode(encoded_input))

plt.figure(figsize=(12, 5))

for i, out in enumerate(outputs):
    outputs[i] = np.array(out).reshape((7, 5))
print(outputs)
sns.heatmap(outputs[1], annot=True, fmt='g')
plt.show()
