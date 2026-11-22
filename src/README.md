## Installation

Dependencies can be installed using uv:
```bash
uv sync --all-groups
# For CUDA users
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/nvidia/cu13/lib"
```

> [!WARNING]
> Although the dependencies are installed, one may still handle CUDA libraries and ffmpeg libraries in OS level.

One can also directly install the dependencies using pip:
```bash
pip install -r requirements.txt
```
In this case, you may need to spend some time to identify the correct path to the CUDA libraries and set the `LD_LIBRARY_PATH` environment variable accordingly.

## Kernel registration

To allow the virtual environment to be used in Jupyter notebooks,
you can register the virtual environment as a Jupyter kernel:
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
The models will be saved in the `model_registry` folder.
We note that the estimated training time is around 3 hours 45 minutes on a single NVIDIA RTX 4060ti 16GB GPU.
The overall training time is estimated 11 hours 15 minutes for 3 random seeds.

## Evaluation

To evaluate the fine-tuned SmolVLA policies, you can run the following command:
```bash
uv run python scripts/evaluate_models.py
```

The script will run each saved checkpoint on 256 evaluation episodes
for each of the 10 visual conditions (clean, gaussian noise, blur, brightness shift)
and 3 severity levels (1, 2, 3).
The evaluation results will be saved in the `model_evaluation` folder.
We note that there are in total 220 evaluation runs,
and each evaluation run takes around 3 minutes on a single NVIDIA RTX 4060ti 16GB GPU.
The overall evaluation time is estimated 11 hours 15 minutes.

## Results

We provide the notebooks to reproduce the reported results in the [src/notebooks](./notebooks) folder.
For details, please refer to the [src/notebooks/README.md](./notebooks//README.md) file.
