"""Semantic reasoning graph construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attributes: Dict[str, Any]


class SemanticReasoningGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: GraphNode) -> None:
        self.graph.add_node(node.node_id, node_type=node.node_type, **node.attributes)

    def add_edge(
        self, source_id: str, target_id: str, relation: str, weight: float = 1.0
    ) -> None:
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            weight=weight,
        )

    def build_graph(self, nodes: List[GraphNode], edges: List[Tuple[str, str, str]]) -> None:
        for node in nodes:
            self.add_node(node)
        for source_id, target_id, relation in edges:
            self.add_edge(source_id, target_id, relation)

    def export_reasoning_trace(self) -> List[Dict[str, Any]]:
        trace = []
        for source, target, data in self.graph.edges(data=True):
            trace.append(
                {
                    "source": source,
                    "target": target,
                    "relation": data.get("relation"),
                    "weight": data.get("weight"),
                }
            )
        return trace

    def to_pyg_data(self) -> Data:
        node_ids = list(self.graph.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        edge_index = []
        edge_attr = []
        for source, target, data in self.graph.edges(data=True):
            edge_index.append([node_index[source], node_index[target]])
            edge_attr.append([float(data.get("weight", 1.0))])
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float)
        x = torch.zeros((len(node_ids), 1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
