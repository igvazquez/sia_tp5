import numpy as np
import pandas as pd
from Autoencoder import Autoencoder

# 7*5
betas = [1, 1, 1, 1, 1, 1, 1, 1, 1]
ae = Autoencoder(35, [30,25,20], 15, betas)

df = pd.read_csv('numerosenbits', delimiter="\n", header=None)

df = np.array(df)
df = df.reshape(10, 7)

remove_spaces = lambda x: x.replace(" ", "")

input_numbers = []
for i in range(len(df)):
    input_numbers.append(remove_spaces("".join(np.squeeze(np.asarray(df[i])))))

for i in range(len(input_numbers)):
    input_numbers[i] = list(input_numbers[i])
    input_numbers[i] = [int(j) for j in input_numbers[i]]

# output_numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 9
output_numbers = input_numbers

ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 10000, 0.8, True)

outputs = []
# new_input = np.zeros(10)
for inp in range(len(input_numbers)):
    # new_input[inp] = 1
    encoded_input = ae.encode(input_numbers[inp])
    outputs.append(ae.decode(encoded_input))
    # new_input[inp] = 0
