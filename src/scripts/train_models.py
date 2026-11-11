import shutil
import subprocess

import numpy as np


seed_seq = np.random.SeedSequence(42)
TRAIN_SEEDS = seed_seq.generate_state(3)

POLICY = "lerobot/smolvla_base"
DATASET_REPO_ID = "lerobot/xarm_lift_medium"
RENAME = {"observation.image": "observation.images.camera1"}
RENAME_STR = str(RENAME)
STEPS = 20_000
SAVE_FREQ = 4_000
BATCH = 32
DEVICE_TYPE = "cuda"
OUT_ROOT = "model_registry"

for train_seed in TRAIN_SEEDS:

    job_name = f"smolvla_{train_seed}"
    train_cmd = [
        shutil.which("lerobot-train"),
        f"--policy.path={POLICY}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--rename_map={RENAME_STR}",
        f"--policy.device={DEVICE_TYPE}",
        f"--steps={STEPS}",
        f"--batch_size={BATCH}",
        f"--save_freq={SAVE_FREQ}",
        f"--seed={train_seed}",
        f"--output_dir={OUT_ROOT}/{job_name}",
        f"--job_name={job_name}",
        "--policy.push_to_hub=false",
        "--env_eval_freq=0",
        "--log_freq=100",
        "--wandb.enable=false",
        "--tolerance_s=0.0005",
    ]

    print(f"Training with seed {train_seed}...")
    subprocess.run(train_cmd, check=True)
