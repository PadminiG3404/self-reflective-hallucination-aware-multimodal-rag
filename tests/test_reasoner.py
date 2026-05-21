import networkx as nx

from src.reasoning.multihop_reasoner import MultiHopReasoner
from src.utils.schemas import EvidenceChunk


def test_reasoner_generates_steps():
    graph = nx.MultiDiGraph()
    graph.add_node("n1")
    graph.add_node("n2")
    evidence = [EvidenceChunk(chunk_id="e1", text="e", source="unit", score=0.0)]
    reasoner = MultiHopReasoner(max_hops=2)
    steps = reasoner.infer(graph, evidence)
    assert len(steps) == 2
