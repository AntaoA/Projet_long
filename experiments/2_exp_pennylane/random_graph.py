import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as pnp
import matplotlib.pyplot as plt


n = 6
p = 0.6
graph = nx.gnp_random_graph(n, p)

# Visualisation du graphe initial
pos = nx.spring_layout(graph)
plt.figure(figsize=(6, 4))
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title(f"Graphe aléatoire")
plt.show()


def find_all_maximum_cliques(G):
    all_maximal = list(nx.find_cliques(G))
    max_size = max(len(c) for c in all_maximal)
    max_cliques = [sorted(c) for c in all_maximal if len(c) == max_size]    
    return max_cliques, max_size

solutions, size = find_all_maximum_cliques(graph)
print(f"Solutions Classiques")
print(f"Taille de la clique maximale : {size}")
print(f"Nombre de solutions trouvées : {len(solutions)}")
print(f"Liste des solutions : {solutions}")

# Visualisation max clique (classique)
plt.figure(figsize=(6, 4))
first_sol = solutions[0]
node_colors = ['red' if n in first_sol else 'lightblue' for n in graph.nodes()]
nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray')
plt.show()



cost_h, mixer_h = qaoa.max_clique(graph, constrained=False)
jitter = pnp.random.uniform(0, 0.001, n)
for i in range(n):
    cost_h += jitter[i] * qml.PauliZ(i)
dev = qml.device("default.qubit", wires=n)

p = 6 
steps = 200 

def qaoa_circuit(params):
    for i in range(n):
        qml.Hadamard(wires=i)
    for i in range(p):
        qml.ApproxTimeEvolution(cost_h, params[0][i], 1)
        qml.ApproxTimeEvolution(mixer_h, params[1][i], 1)

@qml.qnode(dev)
def cost_function(params):
    qaoa_circuit(params)
    return qml.expval(cost_h)

@qml.qnode(dev)
def probability_circuit(params):
    qaoa_circuit(params)
    return qml.probs(wires=range(n))



params = pnp.array([[0.1] * p, [0.1] * p], requires_grad=True)
optimizer = qml.AdamOptimizer(stepsize=0.01) 
cost_history = []

print(f"Lancement de l'optimisation (p={p}, steps={steps})...")
start_time = time.time()

for i in range(steps):
    params, cost = optimizer.step_and_cost(cost_function, params)
    cost_history.append(cost)
    if (i + 1) % 20 == 0:
        print(f"Étape {i+1:3d} | Coût: {cost:10.4f}")

print(f"\nTemps d'exécution : {time.time() - start_time:.2f} secondes")





probs = probability_circuit(params)
states = [format(i, f'0{n}b') for i in range(2**n)]
best_idx = np.argmax(probs)
optimal_bitstring = states[best_idx]
clique_nodes = [i for i, bit in enumerate(optimal_bitstring) if bit == '1']

# Visualisation
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].plot(cost_history, color='navy')
ax[0].set_title("Convergence du Coût")
ax[0].set_xlabel("Itérations")
ax[0].set_ylabel("Coût")

node_colors = ['red' if i in clique_nodes else 'lightblue' for i in range(n)]
nx.draw(graph, pos, with_labels=True, node_color=node_colors, ax=ax[1], node_size=600)
ax[1].set_title(f"Clique: {optimal_bitstring}\nTaille: {len(clique_nodes)}")

top_indices = np.argsort(probs)[-10:][::-1]
top_probs = probs[top_indices]
top_labels = [states[i] for i in top_indices]
ax[2].bar(top_labels, top_probs, color='teal')
ax[2].set_title("Top 10 des configurations")
ax[2].set_ylabel("Probabilité")
plt.setp(ax[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

print(f"Meilleure configuration : {optimal_bitstring}")
print(f"Confiance (Probabilité) : {probs[best_idx]:.4f}")