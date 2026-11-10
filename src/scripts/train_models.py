import shutil
import subprocess

import numpy as np


seed_seq = np.random.SeedSequence(42)
TRAIN_SEEDS = seed_seq.generate_state(3)

RENAME = {"observation.image": "observation.images.camera1"}
RENAME_STR = str(RENAME)
STEPS = 20_000
SAVE_FREQ = 4_000
BATCH = 32
OUT_ROOT = "model_registry"

for train_seed in TRAIN_SEEDS:

    train_cmd = [
        shutil.which("lerobot-train"),
        "--policy.path=lerobot/smolvla_base",
        "--dataset.repo_id=lerobot/xarm_lift_medium",
        f"--rename_map={RENAME_STR}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--steps={STEPS}",
        f"--batch_size={BATCH}",
        "--env_eval_freq=0",
        f"--save_freq={SAVE_FREQ}",
        "--log_freq=100",
        f"--seed={train_seed}",
        "--wandb.enable=false",
        f"--output_dir={OUT_ROOT}/smolvla_{train_seed}",
        f"--job_name=smolvla_{train_seed}",
        "--tolerance_s=0.0005",
    ]

    print(f"Training with seed {train_seed}...")
    subprocess.run(train_cmd, check=True)
