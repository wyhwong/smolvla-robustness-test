import os
import subprocess

import numpy as np


SEED_SEQ = np.random.SeedSequence(42)
TRAIN_SEEDS = SEED_SEQ.generate_state(3)
POLICY_BASE = "lerobot/smolvla_base"
RENAME = {"observation.image": "observation.images.camera1"}
RENAME_STR = str(RENAME)
DEVICE_TYPE = "cuda"
MODEL_REGISTRY_DIR = "model_registry"

EVALUATION_DIR = "model_evaluation"
EVAL_SEED = 42
N_EPISODES = 256
CHECKPOINTS = [
    "004000",
    "008000",
    "012000",
    "016000",
    "020000",
    "last",
]
SCENARIOS = [
    "clean",
]


def get_checkout_path(training_seed: int, checkpoint: str) -> str:

    return f"{MODEL_REGISTRY_DIR}/smolvla_{training_seed}/checkpoints/{checkpoint}/pretrained_model"


def get_evaluation_cmd(
    policy_checkpoint_path: str,
    job_name: str,
    scenario: str,
) -> list[str]:

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get current python executable
    python_executable = subprocess.check_output(["which", "python"]).decode("utf-8").strip()
    print("Python executable found at: %s", python_executable)

    return [
        python_executable,
        f"{script_dir}/evaluation/{scenario}.py",
        f"--policy.path={policy_checkpoint_path}",
        "--env.type=pusht",
        f"--rename_map={RENAME_STR}",
        "--eval.use_async_envs=false",
        f"--eval.batch_size={N_EPISODES}",
        "--policy.device=cuda",
        f"--output_dir={EVALUATION_DIR}/{scenario}/{job_name}",
        f"--seed={EVAL_SEED}",
        f"--eval.n_episodes={N_EPISODES}",
    ]


def main():

    if not os.path.exists(EVALUATION_DIR):
        os.makedirs(EVALUATION_DIR)

    for scenario in SCENARIOS:

        if not os.path.exists(f"{EVALUATION_DIR}/{scenario}"):
            os.makedirs(f"{EVALUATION_DIR}/{scenario}")

        job_name = f"smolvla_base_{EVAL_SEED}"
        eval_cmd = get_evaluation_cmd(POLICY_BASE, job_name, scenario)
        print(f"Evaluating base model with seed {EVAL_SEED}...")
        subprocess.run(
            eval_cmd,
            check=True,
            stdout=open(
                f"{EVALUATION_DIR}/{scenario}/{job_name}_evaluation.log",
                mode="w",
                encoding="utf-8",
            ),
            stderr=subprocess.STDOUT,
        )
        for training_seed in TRAIN_SEEDS:
            for ckpt in CHECKPOINTS:

                ckpt_path = get_checkout_path(training_seed, ckpt)
                job_name = f"smolvla_{training_seed}_{ckpt}_{EVAL_SEED}"
                eval_cmd = get_evaluation_cmd(ckpt_path, job_name, scenario)

                print(
                    f"Evaluating model trained from seed {training_seed} at checkpoint {ckpt} with seed {EVAL_SEED}..."
                )
                subprocess.run(
                    eval_cmd,
                    check=True,
                    stdout=open(
                        f"{EVALUATION_DIR}/{scenario}/{job_name}_evaluation.log",
                        mode="w",
                        encoding="utf-8",
                    ),
                    stderr=subprocess.STDOUT,
                )


if __name__ == "__main__":
    main()
