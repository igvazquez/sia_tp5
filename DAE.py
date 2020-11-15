import numpy as np
import pandas as pd
from Autoencoder import Autoencoder


def randomize(value):
    rand = np.random.rand()
    new_value = 1 if value == 0 else 0
    if rand <= random_prob:

        return new_value
    else:
        return value




df = pd.read_csv('fonttesting.txt', delimiter="\n", header=None, dtype=str)

df = np.array(df)
data = df.reshape(3, 7)
print("df", data)
input_numbers = []
for i in range(len(data)):
    input_numbers.append("".join(np.squeeze(np.asarray(data[i]))))

random_prob = 0.25
random_func = np.vectorize(randomize)
for i in range(len(input_numbers)):
    input_numbers[i] = list(input_numbers[i])
    input_numbers[i] = [int(j) for j in input_numbers[i]]

output_numbers = np.copy(input_numbers)
for i in range(len(input_numbers)):
    input_numbers[i] = random_func(input_numbers[i])

print("input_numbers:", input_numbers)

print("output_numbers: ",output_numbers)

# 7*8 pixeles
hidden_layer = [32, 16,8]
print("shape: ",len(hidden_layer))
betas = np.random.random_sample((1, 2*len(hidden_layer)+2+1))
print("betas:",betas)
ae = Autoencoder(56,hidden_layer , 2, betas[0])
ae.train(np.asarray(input_numbers), np.asarray(output_numbers), 60000, 0.6,True)

# outputs = []
# for inp in range(len(input_numbers)):
#     encoded_input = ae.encode(input_numbers[inp])
#     outputs.append(ae.decode(encoded_input))
