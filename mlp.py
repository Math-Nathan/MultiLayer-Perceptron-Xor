import numpy as np
from ativacoes import sigmoid_bipolar, d_sigmoid_bipolar

class MLP:
    def __init__(self, n_inputs=2, n_hidden=4, n_outputs=1, learning_rate=0.2):
        # Inicialização dos pesos aleatória no intervalo [-1, 1]
        self.W1 = np.random.uniform(-1, 1, (n_inputs, n_hidden))   # Pesos entrada -> oculta
        self.b1 = np.random.uniform(-1, 1, (n_hidden,))            # Bias camada oculta
        self.W2 = np.random.uniform(-1, 1, (n_hidden, n_outputs))  # Pesos oculta -> saída
        self.b2 = np.random.uniform(-1, 1, (n_outputs,))           # Bias saída
        self.lr = learning_rate

    def forward(self, X):
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid_bipolar(self.z1)

        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid_bipolar(self.z2)

        return self.a2

    def backward(self, X, y):
        # Erro na saída (usando derivada da sigmóide bipolar)
        erro_saida = (y - self.a2) * d_sigmoid_bipolar(self.z2)

        # Retropropagação do erro para a camada oculta
        erro_oculta = np.dot(erro_saida, self.W2.T) * d_sigmoid_bipolar(self.z1)

        # Atualização dos pesos da camada de saída
        self.W2 += self.lr * np.outer(self.a1, erro_saida)
        self.b2 += self.lr * erro_saida

        # Atualização dos pesos da camada oculta
        self.W1 += self.lr * np.outer(X, erro_oculta)
        self.b1 += self.lr * erro_oculta