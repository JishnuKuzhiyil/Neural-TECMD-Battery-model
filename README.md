# Neural Equivalent Circuit Models: Universal Differential Equations for Battery Modelling

This repository contains Julia code and data for the article "Neural Equivalent Circuit Models: Universal Differential Equations for Battery Modelling" by Jishnu Ayyangatu Kuzhiyil, Theodoros Damoulas, and W. Dhammika Widanage (2024). 

https://doi.org/10.1016/j.apenergy.2024.123692.

## Repository Contents

- `model_comparison.jl`: Julia code to run and compare the Neural TECMD model with its mechanistic counterpart (TECMD). This script generates plots comparing both models against the experimental dataset.
- `Datasets.jld2`: JLD2 file containing all experimental data.
- `DATA_in_MATLAB_format/`: Folder containing the experimental data in .mat format (MATLAB file format).
- `NN_para_volt.jld2`: File containing parameters of neural network NN_1.
- `NN_para_temp.jld2`: File containing parameters of neural network NN_2.

## Usage

To use this code or data, please run the `model_comparison.jl` script in Julia. This will generate comparison plots between the Neural TECMD and TECMD models using the provided experimental data.

## Citation

If you use the code or data from this repository in your research, please cite our article:

Jishnu Ayyangatu Kuzhiyil, Theodoros Damoulas, W. Dhammika Widanage,
Neural equivalent circuit models: Universal differential equations for battery modelling,
Applied Energy,Volume 371,2024,123692,ISSN 0306-2619,
https://doi.org/10.1016/j.apenergy.2024.123692.
