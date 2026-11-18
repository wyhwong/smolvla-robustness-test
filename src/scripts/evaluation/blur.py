import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature, parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.pipeline import ProcessorStep
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)


@dataclass
class BlurConfig:
    kernel_size: int
    stride = 1


@dataclass
class PerturbedEvalPipelineConfig(EvalPipelineConfig):

    blur: BlurConfig = field(default_factory=BlurConfig)


class AddBlur(ProcessorStep):
    """Custom data processor to add blur to specified keys in the observation."""

    def __init__(self, keys: list[str], kernel_size=3, stride=1):
        self.keys = keys
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Handle observation normalization.
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            for key in self.keys:
                if key in observation:
                    observation[key] = torch.nn.functional.avg_pool2d(
                        observation[key],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    )

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@parser.wrap()
def eval_main(cfg: PerturbedEvalPipelineConfig):
    """
    We note that this is a modified version of `eval_main` from `lerobot/scripts/lerobot_eval.py`.
    This is to allow the evaluation for perturbed observation.
    """

    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(
        "Making environment (batch_size={%d}, async=%s).",
        cfg.eval.batch_size,
        cfg.eval.use_async_envs,
    )

    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware,
    # overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    preprocessor.steps += [
        AddBlur(
            keys=cfg.rename_map.values(),
            kernel_size=cfg.blur.kernel_size,
            stride=cfg.blur.stride,
        )
    ]

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            env_features=cfg.env.features if cfg.eval.recording else None,
            recording_repo_id=cfg.eval.recording_repo_id,
            recording_private=cfg.eval.recording_private,
        )
        logger.info("Overall Aggregated Metrics:")
        logger.info(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            logger.info("\nAggregated Metrics for {%s}:", task_group)
            logger.info(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(
        Path(cfg.output_dir) / "eval_info.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()  # pylint: disable=E1120


if __name__ == "__main__":
    main()
