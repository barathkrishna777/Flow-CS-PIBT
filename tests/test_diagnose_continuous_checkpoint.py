from types import SimpleNamespace

from scripts.diagnose_continuous_checkpoint import resolve_target_velocity_source


def test_resolve_target_velocity_source_uses_checkpoint_config_by_default():
    args = SimpleNamespace(target_velocity_source=None)
    checkpoint = {"dataset_config": {"target_velocity_source": "tracker-preferred"}}

    assert resolve_target_velocity_source(args, checkpoint) == "tracker-preferred"


def test_resolve_target_velocity_source_prefers_cli_override():
    args = SimpleNamespace(target_velocity_source="executed")
    checkpoint = {"dataset_config": {"target_velocity_source": "tracker-preferred"}}

    assert resolve_target_velocity_source(args, checkpoint) == "executed"
