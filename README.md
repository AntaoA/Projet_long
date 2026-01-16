Projet long

Le projet utilise uv. Veuillez l'installer : https://docs.astral.sh/uv/getting-started/installation/

```
git clone https://github.com/AntaoA/Projet_long.git
cd Projet_long
uv sync
```
Vous pourrez ensuite run un fichier python via
```
uv run mon_fichier.py
```

Liste des expérimentations :
- 1 : Test de qiskit
  - first_clique.py : test basique de max clique sur un graphe basique (maison)
  - visual_test.py : deux tests basiques et visualisation des graphes et de la max clique trouvé
  - random_test.py : test un peu plus gros sur un graphe aléatoire (long et pas hyper précis (premier résultat pas toujours le meilleur), voir experiments/1_exp_qiskit/random_graph.png)
- 2 : Test de pennylane
  - random_test.py : même chose que précédemment, mais meilleurs résultats (temps, confiance)
- 3 : Graphe de correspondance
  - corres_graph.py : création du graphe de correspondance
