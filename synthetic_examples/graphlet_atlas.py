"""A class for working with atlases of "graphlets" that can be used as motifs in larger graphs.

The Graphlet class can be constructed either with an input list of graphlets or automatically
considering all nodes up to `max_nodes` nodes via
`GraphletAtlas.connected_graphs_up_to(max_nodes)`. Graphlets can also be aliased for easy
access.

TODO:
  1. Is graphlet degree vector code correct?

"""

import itertools, random, networkx as nx, numpy as np
from networkx.algorithms.isomorphism.isomorph import (
    graph_could_be_isomorphic as isomorphic,
)
from networkx.generators.atlas import graph_atlas_g
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt

def all_connected_subgraphs_containing(G, containing_set, lim, seen=set()):
    """Recursively constructs all connected subgraphs of a graph G containing the passed subgraph.

      `G`: The source graph
      `containing_set`: The set of nodes that must be contained in the output subgraphs
      `lim`: the maximum size of the subgraphs
      `seen`: A recursion helper variable cataloging what sets of nodes have already been searched.

      Returns: `output`, a list of all networkx subgraphs of `G` containing `containing_set` of size
               no more than `lim`.
      Side-effects: Updates `seen` to reflect additional things seen during the search.
    """

    if len(containing_set) == lim: return [containing_set]
    output = []

    neighbors = set([])
    for n in containing_set: neighbors.update(G[n])
    neighbors = neighbors - containing_set

    for n in neighbors:
        new_containing_set = containing_set.union([n])
        if frozenset(new_containing_set) in seen: continue

        seen.update([frozenset(new_containing_set)])
        output.extend(all_connected_subgraphs_containing(G, new_containing_set, lim, seen))

    seen.update([frozenset(containing_set)])
    return output

class GraphletAtlas():
    """A container class for an atlas of graphlets, with utilities for lookup and display."""

    @staticmethod
    def connected_graphs_up_to(max_nodes=5):
        """ Return the atlas of all connected graphs of `max_nodes` nodes or less.
            Attempt to check for isomorphisms and remove.
        """

        assert max_nodes <= 6, f"This will take forever with {max_nodes} nodes"

        U = nx.Graph()
        for G in graph_atlas_g():
            if len(G) > max_nodes: break

            zerodegree = [n for n in G if G.degree(n) == 0]
            for n in zerodegree:
                G.remove_node(n)
            U = nx.disjoint_union(U, G)

        # iterator of graphs of all connected components
        C = (U.subgraph(c) for c in nx.connected_components(U))

        UU = nx.Graph()
        # do quick isomorphic-like check, not a true isomorphism checker
        nlist = []  # list of nonisomorphic graphs
        for G in C:
            if not any(isomorphic(G, G2) for G2 in nlist):
                nlist.append(G)
                UU = nx.disjoint_union(UU, G)  # union the nonisomorphic graphs
        return GraphletAtlas([UU.subgraph(c) for c in nx.connected_components(UU)])

    def _random_sample(self): return random.randint(0, len(self.graphlet_atlas)-1)

    def __init__(
        self,
        graphlet_atlas         = None,
        graphlet_aliases       = None,
    ):
        if graphlet_atlas is None:        graphlet_atlas = atlas(max_nodes=6)
        elif type(graphlet_atlas) is int: graphlet_atlas = atlas(max_nodes=graphlet_atlas)

        if type(graphlet_atlas) is nx.Graph:
            graphlet_atlas = [G.subgraph(c) for c in nx.connected_components(G)]

        if graphlet_aliases is None: graphlet_aliases = {}

        self.graphlet_atlas = graphlet_atlas
        self.graphlet_aliases = {
            'random': self._random_sample,
            **graphlet_aliases,
        }

    def resolve_graphlet_index(self, g):
        if type(g) is int: return g
        elif type(g) is str: return self.graphlet_aliases[g.lower()]()

        raise ValueError(f"Graphlet {g} un-resolvable!")

    def resolve_graphlet(self, g):
        """Identifies the Graphlet associated with the passed object `g`.

            If `g` is a
              * `nx.graph`: returns g,
              * `str`: Looks up `g` in selfs graphlet_aliases and returns the corresponding graphlet
              * `int`: Returns the graphlet at index `g` in `self.graphlet_atlas`.
        """
        if type(g) is nx.Graph: return g

        g_idx = self.resolve_graphlet_index(g)
        assert g_idx < len(self.graphlet_atlas), f"{g_idx} is out of bounds (max {len(self.graphlet_atlas)})"

        G = self.graphlet_atlas[g_idx]

        zerodegree = [n for n in G if G.degree(n) == 0]
        for n in zerodegree:
            G.remove_node(n)
        return g_idx, G

    def display(self):
        """Displays the graphlet according to the `neato` `graphviz` algorithm."""

        print(
            f"Graphlet atlas contains {len(self.graphlet_atlas)} subgraphs of "
            f"up to {max(len(g) for g in self.graphlet_atlas)} nodes:"
        )

        G = nx.disjoint_union_all(self.graphlet_atlas)
        plt.figure(1, figsize=(8, 8))
        # layout graphs with positions using graphviz neato
        pos = graphviz_layout(G, prog="neato")
        # color nodes the same in each connected subgraph
        C = (G.subgraph(c) for c in nx.connected_components(G))
        for g in C:
            c = [random.random()] * nx.number_of_nodes(g)  # random color...
            nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)

    def graphlet_degree_vectors(self, G):
        """Returns a list counting how many times each node in `G` is part of each graphlet in `self.graphlet_atlas.

            This is useful for producing a structural description of the passed graph `G` at the granularity of `self`.
        """
        orbit_counts = [[0 for _ in self.graphlet_atlas] for _ in G]
        valid_subgraph_sizes = set(len(g) for g in self.graphlet_atlas)

        new_connected_subgraphs = [set(x) for x in G.edges()]
        all_connected_subgraphs = new_connected_subgraphs
        for i in range(3, max(valid_subgraph_sizes)+1):
            seen = set()
            new_connected_subgraphs = list(itertools.chain.from_iterable(
                all_connected_subgraphs_containing(G, sg, i, seen=seen) for sg in new_connected_subgraphs
            ))

            if i in valid_subgraph_sizes:
                all_connected_subgraphs.extend(new_connected_subgraphs)

        for sg_nodes in all_connected_subgraphs:
            subgraph = G.subgraph(sg_nodes)
            orbit_idxs = [i for i, gl in enumerate(self.graphlet_atlas) if isomorphic(subgraph, gl)]
            for v in sg_nodes:
                for i in orbit_idxs: orbit_counts[v][i] += 1

        return np.array(orbit_counts)
