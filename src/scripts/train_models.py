import os
import shutil
import subprocess

import numpy as np


SEED_SEQ = np.random.SeedSequence(42)
TRAIN_SEEDS = SEED_SEQ.generate_state(3)
POLICY_BASE = "lerobot/smolvla_base"
DATASET_REPO_ID = "lerobot/pusht"
RENAME = {"observation.image": "observation.images.camera1"}
RENAME_STR = str(RENAME)
N_STEPS = 25_650
CHECKPOINT_SAVE_FREQ = 4_000
BATCH_SIZE = 32
DEVICE_TYPE = "cuda"
MODEL_REGISTRY_DIR = "model_registry"


def main():

    if not os.path.exists(MODEL_REGISTRY_DIR):
        os.makedirs(MODEL_REGISTRY_DIR)

    for train_seed in TRAIN_SEEDS:

        job_name = f"smolvla_{train_seed}"
        output_dir = f"{MODEL_REGISTRY_DIR}/{job_name}"

        train_cmd = [
            shutil.which("lerobot-train"),
            f"--policy.path={POLICY_BASE}",
            f"--dataset.repo_id={DATASET_REPO_ID}",
            f"--rename_map={RENAME_STR}",
            f"--policy.device={DEVICE_TYPE}",
            f"--steps={N_STEPS}",
            f"--batch_size={BATCH_SIZE}",
            f"--save_freq={CHECKPOINT_SAVE_FREQ}",
            f"--seed={train_seed}",
            f"--output_dir={output_dir}",
            f"--job_name={job_name}",
            "--policy.push_to_hub=false",
            "--env_eval_freq=0",
            "--log_freq=100",
            "--wandb.enable=false",
        ]

        print(f"Training with seed {train_seed}...")
        subprocess.run(
            train_cmd,
            check=True,
            stdout=open(
                f"{MODEL_REGISTRY_DIR}/{job_name}_training.log",
                mode="w",
                encoding="utf-8",
            ),
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
