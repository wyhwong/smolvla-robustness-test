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
    "024000",
    "last",
]
SCENARIOS = [
    "clean",
    "gaussian_noise",
    "blur",
    "brightness_shift",
]


def get_checkout_path(training_seed: int, checkpoint: str) -> str:

    return f"{MODEL_REGISTRY_DIR}/smolvla_{training_seed}/checkpoints/{checkpoint}/pretrained_model"


def get_evaluation_cmd(
    policy_checkpoint_path: str,
    job_name: str,
    scenario: str,
    extra_args: list[str] | None = None,
) -> list[str]:

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get current python executable
    python_executable = subprocess.check_output(["which", "python"]).decode("utf-8").strip()
    print(f"Python executable found at: {python_executable}")

    cmd = [
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

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def main():

    if not os.path.exists(EVALUATION_DIR):
        os.makedirs(EVALUATION_DIR)

    extra_args_by_scenario = {
        "clean": {
            "default": None,
        },
        "gaussian_noise": {
            "severity_level_1": ["--gaussian_noise.std=0.03"],
            "severity_level_2": ["--gaussian_noise.std=0.1"],
            "severity_level_3": ["--gaussian_noise.std=0.3"],
        },
        "blur": {
            "severity_level_1": ["--blur.kernel_size=3"],
            "severity_level_2": ["--blur.kernel_size=5"],
            "severity_level_3": ["--blur.kernel_size=7"],
        },
        "brightness_shift": {
            "severity_level_1": ["--brightness_shift.shift=-0.3"],
            "severity_level_2": ["--brightness_shift.shift=0.05"],
            "severity_level_3": ["--brightness_shift.shift=0.3"],
        },
    }

    for scenario in SCENARIOS:
        for severity_level, extra_args in extra_args_by_scenario[scenario].items():
            if not os.path.exists(f"{EVALUATION_DIR}/{scenario}"):
                os.makedirs(f"{EVALUATION_DIR}/{scenario}")

            job_name = f"smolvla_base_{scenario}_{severity_level}_{EVAL_SEED}"
            eval_cmd = get_evaluation_cmd(
                policy_checkpoint_path=POLICY_BASE,
                job_name=job_name,
                scenario=scenario,
                extra_args=extra_args,
            )
            if not os.path.exists(f"{EVALUATION_DIR}/{scenario}/{job_name}/eval_info.json"):
                print(f"Evaluating base model {scenario} {severity_level} with seed {EVAL_SEED}...")
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
            else:
                print(
                    "Found existing evaluation for base model "
                    f"{scenario} {severity_level} with seed {EVAL_SEED}, skipping..."
                )

            for training_seed in TRAIN_SEEDS:
                for ckpt in CHECKPOINTS:

                    ckpt_path = get_checkout_path(
                        training_seed=training_seed,
                        checkpoint=ckpt,
                    )
                    job_name = f"smolvla_{training_seed}_{ckpt}_{scenario}_{severity_level}_{EVAL_SEED}"
                    eval_cmd = get_evaluation_cmd(
                        policy_checkpoint_path=ckpt_path,
                        job_name=job_name,
                        scenario=scenario,
                        extra_args=extra_args,
                    )

                    if os.path.exists(f"{EVALUATION_DIR}/{scenario}/{job_name}/eval_info.json"):
                        print(
                            f"Found existing evaluation for model in {scenario} {severity_level} trained "
                            f"from seed {training_seed} at checkpoint {ckpt} with seed {EVAL_SEED}, skipping..."
                        )
                        continue

                    print(
                        f"Evaluating model in {scenario} {severity_level} trained from seed {training_seed}"
                        f" at checkpoint {ckpt} with seed {EVAL_SEED}..."
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
