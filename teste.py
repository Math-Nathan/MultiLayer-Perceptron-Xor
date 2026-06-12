import numpy as np
import matplotlib.pyplot as plt
from ativacoes import sigmoid_binary, sigmoid_bipolar, tanh

x = np.linspace(-5, 5, 100)
plt.plot(x, sigmoid_binary(x), label="Sigmóide Binária")
plt.plot(x, sigmoid_bipolar(x), label="Sigmóide Bipolar")
plt.plot(x, tanh(x), label="Tangente Hiperbólica")
plt.legend()
plt.grid()
plt.show()