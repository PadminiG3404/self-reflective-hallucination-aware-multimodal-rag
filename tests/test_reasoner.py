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


def test_reasoner_ranks_paths():
    graph = nx.MultiDiGraph()
    graph.add_node("query")
    graph.add_node("n1", score=0.8)
    graph.add_node("n2", score=0.1)
    graph.add_edge("query", "n1", weight=0.9, relation="retrieval")
    graph.add_edge("query", "n2", weight=0.2, relation="retrieval")
    evidence = [EvidenceChunk(chunk_id="e1", text="node one", source="unit", score=0.8)]
    reasoner = MultiHopReasoner(max_hops=2)
    ranked = reasoner.rank_candidate_paths(graph, evidence)
    assert ranked
    assert ranked[0][0][-1] == "n1"
