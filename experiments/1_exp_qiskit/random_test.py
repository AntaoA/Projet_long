import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Clique

n = 6
p = 0.8
graph = nx.gnp_random_graph(n, p)

# Visualisation du graphe initial
pos = nx.spring_layout(graph)
plt.figure(figsize=(6, 4))
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Graphe aléatoire")
plt.show()


clique_app = Clique(graph)
prob = clique_app.to_quadratic_program()
print("Problème QUBO généré :")
print(prob.export_as_lp_string())


sampler = StatevectorSampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=5)
qe_optimizer = MinimumEigenOptimizer(qaoa)

start_time = time.time()
result = qe_optimizer.solve(prob)
end_time = time.time()

print(f"Temps d'exécution avec SPSA : {end_time - start_time:.2f} secondes")
print(f"Vecteur solution : {result.x}")
print(f"Taille de la clique trouvée : {result.fval}")


optimal_bitstring = "".join(map(str, result.x.astype(int)))
all_samples = sorted(result.samples, key=lambda x: x.probability, reverse=True)
sol_sample = next((s for s in all_samples if "".join(map(str, s.x.astype(int))) == optimal_bitstring), None)
sol_prob = sol_sample.probability if sol_sample else 0
sol_rank = next(i for i, s in enumerate(all_samples) if "".join(map(str, s.x.astype(int))) == optimal_bitstring) + 1

# Visualisation de la clique sur le graphe
clique_nodes = clique_app.interpret(result)
node_colors = ['red' if i in clique_nodes else 'lightblue' for i in range(n)]
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=600)
plt.title(f"Clique Maximale (Taille: {len(clique_nodes)})\nSolution: {optimal_bitstring}")

# Visualisation de la répartition (Top 10)
top_10_samples = all_samples
labels = ["".join(map(str, s.x.astype(int))) for s in top_10_samples]
probs = [s.probability for s in top_10_samples]
colors = ['red' if label == optimal_bitstring else 'teal' for label in labels]
plt.subplot(1, 2, 2)
bars = plt.bar(labels, probs, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Probabilité")
plt.xlabel("Bitstrings (Configurations)")
plt.title(f"Distribution - Probabilité de la solution : {sol_prob:.4f}\n(Rang : {sol_rank} / {len(all_samples)})")

if optimal_bitstring in labels:
    idx = labels.index(optimal_bitstring)
    plt.annotate('Solution choisie', xy=(idx, probs[idx]), xytext=(idx, probs[idx] + 0.0005),
                 ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.title(f"Temps d'exécusion : {end_time - start_time:.2f} secondes")
plt.show()

print(f"Nœuds de la clique : {clique_nodes}")
print(f"Position dans la distribution : {sol_rank}ème")
print(f"Probabilité de mesure : {sol_prob}")