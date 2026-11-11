## Installation

Dependencies can be installed using uv:
```bash
uv sync --all-groups
# For CUDA users
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/nvidia/cu13/lib"
```

## Kernel registration

```bash
# Register the virtual environment as a Jupyter kernel
uv run python -m ipykernel install --user \
  --name smolvla-robustness-test \
  --display-name "Python smolvla robustness test"

# Unregister can do
# uv run jupyter kernelspec remove smolvla-robustness-test

# Find the path to the kernel.json file
jupyter kernelspec list

# Add the following to the kernel.json
# "env": {
#     "LD_LIBRARY_PATH": "<path_to_venv>/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
# }
```

## First test on environment

With the dependencies installed, you can test if the environment is set up correctly by running the following command:
```bash
uv run python -c "import torch; import torchcodec; print(torch.__version__, torch.cuda.is_available(), torchcodec.__version__)"
```

If you have CUDA capable GPU, you should see something like:
```
2.11.0+cu130 True 0.11.1+cu130
```

## Training

To prepare the fine-tuned SmolVLA policies, you can run the following command:
```bash
uv run python scripts/train_models.py
```
Then the models will be saved in the `model_registry` folder.
We note that the estimated training time is around 3 hours on a single NVIDIA RTX 4060ti 16GB GPU.
