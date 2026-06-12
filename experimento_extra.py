import os
os.makedirs("graficos", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# ─── Dados XOR bipolares ──────────────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[-1],[1],[1],[-1]])

MAX_EPOCAS = 10000
TOLERANCIA = 0.001

def treinar(n_hidden=4, lr=0.2, seed=0):
    np.random.seed(seed)
    rede = MLP(n_inputs=2, n_hidden=n_hidden, n_outputs=1, learning_rate=lr)
    erros = []
    for epoca in range(MAX_EPOCAS):
        erro_total = 0.0
        for i in range(len(X)):
            saida = rede.forward(X[i])
            rede.backward(X[i], y[i])
            erro_total += np.mean((y[i] - saida)**2)
        erros.append(erro_total)
        if erro_total < TOLERANCIA:
            return epoca + 1, erro_total, erros
    return MAX_EPOCAS, erros[-1], erros


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO A — Quantidade de neurônios na camada intermediária
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("EXPERIMENTO A — Neurônios na camada intermediária")
print("=" * 55)
print(f"{'Neurônios':<12} {'Épocas':<12} {'Erro Final'}")
print("-" * 40)

configs_a  = [2, 3, 4, 5]
resultados_a = {}
fig_a, ax_a = plt.subplots(figsize=(9, 5))

for n in configs_a:
    epocas, erro, erros = treinar(n_hidden=n, lr=0.2, seed=42)
    resultados_a[n] = (epocas, erro)
    print(f"{n:<12} {epocas:<12} {erro:.6f}")
    ax_a.plot(erros, label=f"{n} neurônios ({epocas} épocas)")

ax_a.set_title("Experimento A — Erro × Época por nº de neurônios ocultos")
ax_a.set_xlabel("Época")
ax_a.set_ylabel("Erro Quadrático Total")
ax_a.legend()
ax_a.grid(True)
plt.tight_layout()
plt.savefig("graficos/experimento_a.png", dpi=150)
plt.show()
print("→ Gráfico salvo: graficos/experimento_a.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO B — Taxa de aprendizagem (fixando 4 neurônios)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("EXPERIMENTO B — Taxa de aprendizagem (4 neurônios)")
print("=" * 55)
print(f"{'Alpha (α)':<12} {'Épocas':<12} {'Erro Final'}")
print("-" * 40)

taxas = [0.1, 0.2, 0.3, 0.4, 0.5]
resultados_b = {}
fig_b, ax_b = plt.subplots(figsize=(9, 5))

for lr in taxas:
    epocas, erro, erros = treinar(n_hidden=4, lr=lr, seed=42)
    resultados_b[lr] = (epocas, erro)
    conv = f"{epocas}" if epocas < MAX_EPOCAS else f"{epocas} (não conv.)"
    print(f"{lr:<12} {conv:<12} {erro:.6f}")
    ax_b.plot(erros, label=f"α={lr} ({epocas} épocas)")

ax_b.set_title("Experimento B — Erro × Época por taxa de aprendizagem")
ax_b.set_xlabel("Época")
ax_b.set_ylabel("Erro Quadrático Total")
ax_b.legend()
ax_b.grid(True)
plt.tight_layout()
plt.savefig("graficos/experimento_b.png", dpi=150)
plt.show()
print("→ Gráfico salvo: graficos/experimento_b.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO C — Influência da inicialização dos pesos (seeds)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("EXPERIMENTO C — Inicialização dos pesos (5 seeds)")
print("=" * 55)
print(f"{'Seed':<10} {'Épocas':<12} {'Erro Final':<15} {'Convergiu?'}")
print("-" * 50)

seeds = [0, 1, 2, 3, 4]
resultados_c = {}
fig_c, ax_c = plt.subplots(figsize=(9, 5))

for seed in seeds:
    epocas, erro, erros = treinar(n_hidden=4, lr=0.2, seed=seed)
    convergiu = "Sim" if epocas < MAX_EPOCAS else "Não"
    resultados_c[seed] = (epocas, erro, convergiu)
    print(f"{seed:<10} {epocas:<12} {erro:<15.6f} {convergiu}")
    ax_c.plot(erros, label=f"seed={seed} ({epocas} épocas)")

ax_c.set_title("Experimento C — Erro × Época por semente aleatória")
ax_c.set_xlabel("Época")
ax_c.set_ylabel("Erro Quadrático Total")
ax_c.legend()
ax_c.grid(True)
plt.tight_layout()
plt.savefig("graficos/experimento_c.png", dpi=150)
plt.show()
print("→ Gráfico salvo: graficos/experimento_c.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("RESUMO — Experimento A (melhor config. de neurônios)")
print("=" * 55)
melhor_a = min(resultados_a, key=lambda k: resultados_a[k][0])
print(f"Melhor: {melhor_a} neurônios → {resultados_a[melhor_a][0]} épocas\n")

print("=" * 55)
print("RESUMO — Experimento B (melhor taxa de aprendizado)")
print("=" * 55)
convs_b = {k: v for k, v in resultados_b.items() if v[0] < MAX_EPOCAS}
if convs_b:
    melhor_b = min(convs_b, key=lambda k: convs_b[k][0])
    print(f"Melhor: α={melhor_b} → {convs_b[melhor_b][0]} épocas\n")
else:
    print("Nenhuma taxa convergiu.\n")

print("=" * 55)
print("RESUMO — Experimento C (estabilidade das seeds)")
print("=" * 55)
conv_c = [v for v in resultados_c.values() if v[2] == "Sim"]
print(f"Seeds que convergiram: {len(conv_c)}/5")
if conv_c:
    media = np.mean([v[0] for v in conv_c])
    print(f"Média de épocas (convergidas): {media:.0f}")