# Notebooks

This directory contains notebooks for demonstrating visual perturbations and analyzing SmolVLA robustness results.

Run the notebooks from this directory so that their relative paths resolve correctly.

## `1_visual_perturbation_demo.ipynb`

This notebook demonstrates how the evaluated visual perturbations are applied to a sample PushT image.
It loads the `lerobot/pusht` dataset and generates example images for Gaussian noise, blur, and brightness shifts,
saving the visualizations under `../model_evaluation/perturbation_examples/`.

## `2_successful_rate.ipynb`

This notebook aggregates evaluation metadata across training seeds and checkpoints for clean and perturbed scenarios.
It calculates success rates using a maximum-reward threshold of 0.8
and writes the combined results to `../model_evaluation/summary/success_rates.json`.

Note that this notebook requires the evaluation results to be present in the `../model_evaluation/` folder, which is prepared by running the `evaluate_models.py` script.

## `3_visualization.ipynb`

Loads the computed success-rate summary and visualizes robustness across training checkpoints.
It produces success-rate curves, clean-versus-perturbed degradation plots,
and a final-checkpoint comparison for Gaussian noise, blur, and brightness shifts.
Generated figures are saved under `../model_evaluation/summary/`.
