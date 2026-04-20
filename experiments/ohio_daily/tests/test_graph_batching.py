import torch

from devp_ohio.datasets import _collate_graph_samples


def test_collate_graph_samples_offsets_edges() -> None:
    sample_a = {
        "dynamic": torch.randn(3, 2, 5),
        "static": torch.randn(2, 4),
        "target": torch.randn(2),
        "target_raw": torch.randn(2),
        "wb_precip": torch.randn(2),
        "wb_pet": torch.randn(2),
        "wb_swi_prev": torch.rand(2),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_weight": torch.ones(1),
        "node_ids": ["a0", "a1"],
        "basin_id_per_node": ["A", "A"],
        "date_per_node": ["2001-01-01", "2001-01-01"],
        "is_outlet_mask": torch.tensor([False, True]),
        "usgs_per_node": torch.tensor([float("nan"), 1.5]),
    }
    sample_b = {
        "dynamic": torch.randn(3, 3, 5),
        "static": torch.randn(3, 4),
        "target": torch.randn(3),
        "target_raw": torch.randn(3),
        "wb_precip": torch.randn(3),
        "wb_pet": torch.randn(3),
        "wb_swi_prev": torch.rand(3),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_weight": torch.ones(2),
        "node_ids": ["b0", "b1", "b2"],
        "basin_id_per_node": ["B", "B", "B"],
        "date_per_node": ["2001-01-02"] * 3,
        "is_outlet_mask": torch.tensor([False, False, True]),
        "usgs_per_node": torch.tensor([float("nan"), float("nan"), 2.5]),
    }

    batch = _collate_graph_samples([sample_a, sample_b])
    assert batch["dynamic"].shape == (3, 5, 5)
    assert batch["static"].shape == (5, 4)
    assert batch["edge_index"].shape[1] == 3
    assert torch.equal(batch["edge_index"][:, 1:], torch.tensor([[2, 3], [3, 4]], dtype=torch.long))
    assert batch["num_graphs"] == 2
