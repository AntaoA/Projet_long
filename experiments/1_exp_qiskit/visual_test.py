import networkx as nx
import matplotlib.pyplot as plt
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Clique

def run_visual_test(g, title):
    clique_app = Clique(g)
    qp = clique_app.to_quadratic_program()
    sampler = StatevectorSampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=3)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    clique_indices = clique_app.interpret(result)
    
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 5))
    nx.draw(g, pos, with_labels=True, node_color='lightgrey', node_size=800)
    nx.draw_networkx_nodes(g, pos, nodelist=clique_indices, node_color='r', node_size=800)
    plt.title(f"{title}\nClique prédite : {clique_indices}")
    plt.show()

# Test : graphe Maison
g_house = nx.house_graph() 
run_visual_test(g_house, "Test A : Graphe Maison")

# Test : triangle vs segment
g_disjoint = nx.Graph()
g_disjoint.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4)]) 
#run_visual_test(g_disjoint, "Test B : Triangle vs Segment")