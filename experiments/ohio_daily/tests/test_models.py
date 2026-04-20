import torch

from devp_ohio.models import HybridSeqGConvSeq2One, LumpedSeq2One


def test_lumped_forward_shape() -> None:
    model = LumpedSeq2One(dynamic_dim=5, static_dim=7, hidden_dim=32, embedding_dim=8, dropout=0.0)
    dynamic = torch.randn(4, 30, 5)
    static = torch.randn(4, 7)
    output = model(dynamic, static)
    assert set(output) == {"discharge", "swi"}
    assert output["discharge"].shape == (4,)
    assert output["swi"].shape == (4,)
    assert torch.all((0.0 <= output["swi"]) & (output["swi"] <= 1.0))


def test_graph_forward_shape() -> None:
    model = HybridSeqGConvSeq2One(dynamic_dim=5, static_dim=7, hidden_dim=32, embedding_dim=8, dropout=0.0)
    dynamic = torch.randn(30, 6, 5)
    static = torch.randn(6, 7)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    output = model(dynamic, static, edge_index, edge_weight)
    assert set(output) == {"discharge", "swi"}
    assert output["discharge"].shape == (6,)
    assert output["swi"].shape == (6,)
    assert torch.all((0.0 <= output["swi"]) & (output["swi"] <= 1.0))
