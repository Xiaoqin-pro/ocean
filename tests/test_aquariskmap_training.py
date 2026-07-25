import torch

from reliability.aquariskmap import AquaRiskMap, atomic_torch_save
from scripts.train_aquariskmap_pilot import checkpoint_payload, load_completed_epoch_checkpoint, smoke_is_complete


def test_aquariskmap_checkpoint_round_trip_restores_completed_epoch(tmp_path):
    model = AquaRiskMap(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"
    atomic_torch_save(checkpoint_payload(2, 19, model, optimizer, scaler, model_name="segformer", positive_weight=3.0), path)
    restored = AquaRiskMap(); restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert load_completed_epoch_checkpoint(path, restored, restored_optimizer, restored_scaler, torch.device("cpu")) == (3, 19, 3.0)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_completed_smoke_resume_does_not_run_an_extra_step():
    assert smoke_is_complete(100, 100)
    assert smoke_is_complete(101, 100)
    assert not smoke_is_complete(99, 100)
    assert not smoke_is_complete(0, 0)
