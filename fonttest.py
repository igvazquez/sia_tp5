import numpy as np
import pandas as pd
from Autoencoder import Autoencoder

ae = Autoencoder(35, [20, 15], 10)

df = pd.read_csv('font2.txt', delimiter="\n", header=None, dtype=str)

df = np.array(df)
data = df.reshape(32, 7)
test = data[1]

input_numbers = []
for i in range(len(data)):
    input_numbers.append("".join(np.squeeze(np.asarray(data[i]))))

for i in range(len(input_numbers)):
    input_numbers[i] = list(input_numbers[i])
    input_numbers[i] = [int(j) for j in input_numbers[i]]

output_numbers = input_numbers

# 7*8 pixeles
ae = Autoencoder(56, [100, 50, 40], 32)
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 10000, 0.5, True)

outputs = []
for inp in range(len(input_numbers)):
    encoded_input = ae.encode(input_numbers[inp])
    outputs.append(ae.decode(encoded_input))
