import os
os.makedirs("graficos", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# ─── Dados XOR (representação bipolar: -1 no lugar de 0) ──────────────────────
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[-1],   # 0 XOR 0 = 0  → -1
              [ 1],   # 0 XOR 1 = 1  → +1
              [ 1],   # 1 XOR 0 = 1  → +1
              [-1]])  # 1 XOR 1 = 0  → -1

# ─── Parâmetros (conforme artigo) ─────────────────────────────────────────────
MAX_EPOCAS  = 10000
TOLERANCIA  = 0.001
LR          = 0.2
N_HIDDEN    = 4

# ─── Treinamento ──────────────────────────────────────────────────────────────
np.random.seed(42)
rede = MLP(n_inputs=2, n_hidden=N_HIDDEN, n_outputs=1, learning_rate=LR)

erros          = []
erros_absolutos = []
epocas_convergencia = MAX_EPOCAS

for epoca in range(MAX_EPOCAS):
    erro_total    = 0.0
    erro_absoluto = 0.0

    for i in range(len(X)):
        saida       = rede.forward(X[i])
        rede.backward(X[i], y[i])
        erro_total    += np.mean((y[i] - saida) ** 2)
        erro_absoluto += np.abs(y[i] - saida).sum()

    erros.append(erro_total)
    erros_absolutos.append(erro_absoluto)

    # Critério de parada
    if erro_total < TOLERANCIA:
        epocas_convergencia = epoca + 1
        print(f"✔ Convergiu na época {epocas_convergencia} | Erro: {erro_total:.6f}")
        break
else:
    print(f"⚠ Não convergiu em {MAX_EPOCAS} épocas | Erro final: {erros[-1]:.6f}")

# ─── Saídas finais ────────────────────────────────────────────────────────────
print("\nSaídas finais da rede:")
print(f"{'Entrada':<15} {'Saída bruta':<15} {'Saída arredondada'}")
for i in range(len(X)):
    saida_bruta = rede.forward(X[i])[0]
    saida_arred = 1 if saida_bruta >= 0 else -1
    print(f"{str(X[i]):<15} {saida_bruta:<15.4f} {saida_arred}")

# ─── Gráfico Erro × Época ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(erros, color='crimson')
axes[0].set_title("Erro Quadrático × Época")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Erro Quadrático Total")
axes[0].grid(True)

axes[1].plot(erros_absolutos, color='steelblue')
axes[1].set_title("Erro Absoluto Total × Época")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Erro Absoluto Total")
axes[1].grid(True)

plt.suptitle(f"Treinamento XOR — {epocas_convergencia} épocas | α={LR} | {N_HIDDEN} neurônios ocultos")
plt.tight_layout()
plt.savefig("graficos/resultado_xor.png", dpi=150)
plt.show()
print("Gráfico salvo em graficos/resultado_xor.png")