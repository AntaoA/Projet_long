import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FILE_1 = 'data/plastocyanin_ca.ods'                      # ou data/chymotrypsin_ca.ods
FILE_2 = 'data/azurin_ca.ods'                            # ou data/subtilisin_ca.ods
EPSILON = 0.7   # Tolérance géométrique (Å)
D_MAX = 10.0    # Rayon local serré pour le site actif (Å)

NUM_CORES = -1

def load_all_atoms(file):
    cols = ['Serial', 'Atom', 'Res', 'Seq', 'X', 'Y', 'Z']
    df = pd.read_excel(file, engine='odf', header=None, names=cols)
    return df.reset_index(drop=True)

df_1 = load_all_atoms(FILE_1)
df_2 = load_all_atoms(FILE_2)
coords_1 = df_1[['X', 'Y', 'Z']].values
coords_2 = df_2[['X', 'Y', 'Z']].values

print(f"Protéine 1 : {len(df_1)} résidus, protéine 2 : {len(df_2)}")

dist_matrix_1 = cdist(coords_1, coords_1)
dist_matrix_2 = cdist(coords_2, coords_2)


# Nœuds identité stricte
nodes_id = []
for i, row_i in df_1.iterrows():
    for j, row_j in df_2.iterrows():
        if row_i['Res'] == row_j['Res']:
            nodes_id.append((i, j))

# Nœuds identité chimique
ch_map = {
    'ALA': 'hydrophobic', 'VAL': 'hydrophobic', 'LEU': 'hydrophobic', 'ILE': 'hydrophobic',
    'PHE': 'hydrophobic', 'TRP': 'hydrophobic', 'MET': 'hydrophobic', 'PRO': 'hydrophobic', 'GLY': 'hydrophobic',
    'SER': 'polar', 'THR': 'polar', 'CYS': 'polar', 'TYR': 'polar', 'ASN': 'polar', 'GLN': 'polar',
    'ASP': 'acidic', 'GLU': 'acidic',
    'LYS': 'basic', 'ARG': 'basic', 'HIS': 'basic'
}
nodes_ch = []
for i, row_i in df_1.iterrows():
    for j, row_j in df_2.iterrows():
        res_i = row_i['Res']
        res_j = row_j['Res']
        if ch_map.get(res_i) == ch_map.get(res_j):
            nodes_ch.append((i, j))

# Nœuds tout autorisé
nodes_all = []
nodes_all = [(i, j) for i in df_1.index for j in df_2.index]

print(f"Nombre de nœuds :")
print(f"\tSans filtre : {len(nodes_all)}")
print(f"\tFiltre propriété : {len(nodes_ch)}")
print(f"\tFiltre strict: {len(nodes_id)}")


# Choix filtre
nodes = nodes_id

nodes.sort() 
n_nodes = len(nodes)
G_corr = nx.Graph()
G_corr.add_nodes_from(nodes)



# Arêtes
def compute_edges(start_idx, end_idx, nodes_list, dist_1, dist_2, eps, d_max):
    local_edges = []
    for idx_u in range(start_idx, end_idx):
        u = nodes_list[idx_u]
        i, j = u
        for idx_v in range(idx_u + 1, len(nodes_list)):
            v = nodes_list[idx_v]
            k, l = v
            if i >= k or j >= l: 
                continue
            d_ik = dist_1[i, k]
            if d_ik > d_max: continue
            d_jl = dist_2[j, l]
            if abs(d_ik - d_jl) <= eps:
                local_edges.append((u, v))
    return local_edges


chunks = np.array_split(range(n_nodes), min(n_nodes, 100)) 
results = Parallel(n_jobs=NUM_CORES)(
    delayed(compute_edges)(c[0], c[-1]+1, nodes, dist_matrix_1, dist_matrix_2, EPSILON, D_MAX) 
    for c in tqdm(chunks)
)

for edge_list in results:
    if edge_list:
        G_corr.add_edges_from(edge_list)


print(f"Avant élagage :")
print(f"\tNœuds : {G_corr.number_of_nodes()}")
print(f":\tArêtes : {G_corr.number_of_edges()}")

# Élagage
elag_size = 4                           # élagage par alag_size-clique
nodes_to_keep = set()
for clique in nx.find_cliques(G_corr):
    if len(clique) >= elag_size:
        nodes_to_keep.update(clique)

G_final = G_corr.subgraph(nodes_to_keep).copy()

print(f"Graphe final :")
print(f"\tNœuds : {G_final.number_of_nodes()}")
print(f"\tArêtes : {G_final.number_of_edges()}")




plt.figure(figsize=(10, 8))

pos = nx.spring_layout(G_final)
color_map_res = {
    'HIS': '#1f77b4', 'CYS': '#ff7f0e', 'MET': '#2ca02c', 
    'ASP': '#d62728', 'SER': '#9467bd', 'GLY': '#8c564b',
    'ALA': '#e377c2', 'THR': '#7f7f7f', 'PHE': '#bcbd22', 'TYR': '#17becf'
}
default_color = '#aec7e8'

labels = {}
node_colors = []
res_present = set()

for node in G_final.nodes():
    idx_1, idx_2 = node
    res_name = df_1.iloc[idx_1]['Res']
    labels[node] = f"{df_1.iloc[idx_1]['Seq']}-{df_2.iloc[idx_2]['Seq']}"
    
    color = color_map_res.get(res_name, default_color)
    node_colors.append(color)
    res_present.add(res_name if res_name in color_map_res else 'Autre')

nx.draw_networkx_edges(G_final, pos, alpha=0.4, edge_color='#555555', width=1.2)
nx.draw_networkx_nodes(
    G_final, pos, 
    node_size=120, 
    node_color=node_colors,
    edgecolors='white', 
    linewidths=0.8
)

pos_labels = {k: (v[0], v[1] + 0.025) for k, v in pos.items()}
nx.draw_networkx_labels(
    G_final, pos_labels, 
    labels=labels, 
    font_size=7, 
    font_weight='bold',
    font_family='sans-serif'
)

legend_patches = [mpatches.Patch(color=color_map_res[res], label=res) for res in color_map_res if res in res_present]
if 'Autre' in res_present:
    legend_patches.append(mpatches.Patch(color=default_color, label='Autre'))

plt.legend(
    handles=legend_patches, 
    title="Acides Aminés", 
    loc='upper right', 
    bbox_to_anchor=(1.1, 1),
    frameon=True,
    shadow=True
)

plt.title(f"Graphe de correspondance \n$\epsilon$={EPSILON}Å, $D_{{MAX}}$={D_MAX}Å", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()



# clique maximales
print("La clique maximale dans le graphe de correspondance a une taille de :",
      max(len(c) for c in nx.find_cliques(G_final)))

print("Liste des cliques maximales :")
for c in nx.find_cliques(G_final):
    if len(c) ==  max(len(c) for c in nx.find_cliques(G_final)):
        print(c)
