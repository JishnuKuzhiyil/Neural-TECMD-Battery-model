# Neural Equivalent Circuit Models : Universal Differential Equations for Battery Modelling

Julia code and data for the article "Neural Equivalent Circuit Models : Universal Differential Equations for Battery Modelling" by Jishnu Ayyangatu Kuzhiyil, Theodoros Damoulas, and 
W. Dhammika Widanage (2024).

## What is in this repository ?

In this repository, you can find julia code to run and comapre Neural TECMD model with its mechanistic counterpart (TECMD).
The julia code in "model_comparison.jl", gives plots comparing both models against experimental dataset. The JLD2 file "Datasets.jl"
has all the experiumental data. The files "NN_para_volt.jld2" and "NN_para_temp.jld2" contain parameters of neural network NN_1 and NN_2 respectively.  
