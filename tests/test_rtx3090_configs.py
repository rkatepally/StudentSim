from pathlib import Path

from studentsim.training.config import TrainingConfig


CONFIG_DIR = Path("configs/training/rtx3090")


def test_all_rtx3090_configs_are_single_gpu_and_memory_conservative():
    paths = sorted(CONFIG_DIR.glob("stage*.yaml"))
    assert len(paths) == 6

    for path in paths:
        config = TrainingConfig.from_yaml(path)
        assert config.world_size == 1
        assert config.per_device_batch == 1
        assert config.effective_batch == config.gradient_accumulation
        assert config.max_seq_len <= 2048
        assert config.lora.rank <= 32
        assert config.lora.alpha <= 64


def test_stage2_configs_continue_from_matching_rtx3090_stage1_adapter():
    for path in sorted(CONFIG_DIR.glob("stage2_*.yaml")):
        config = TrainingConfig.from_yaml(path)
        assert config.initial_adapter is not None
        assert "/rtx3090/stage1/adapter" in config.initial_adapter
