import networkx as nx
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Clique


g = nx.Graph()
g.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])

clique_app = Clique(g)
qp = clique_app.to_quadratic_program()

sampler = StatevectorSampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=2)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
clique_indices = clique_app.interpret(result)

print(f"Indices de la clique maximale : {clique_indices}")