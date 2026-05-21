from src.graph_reasoning.semantic_graph import GraphNode, SemanticReasoningGraph


def test_graph_build_and_export():
    graph = SemanticReasoningGraph()
    nodes = [
        GraphNode(node_id="n1", node_type="entity", attributes={"label": "A"}),
        GraphNode(node_id="n2", node_type="entity", attributes={"label": "B"}),
    ]
    edges = [("n1", "n2", "relation")]
    graph.build_graph(nodes, edges)
    trace = graph.export_reasoning_trace()
    assert len(trace) == 1
    pyg_data = graph.to_pyg_data()
    assert pyg_data.edge_index.shape[1] == 1
