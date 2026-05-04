Welcome to the repo for Ross Kempner's NDS final project

I tested this on a linux machine with vscode.

Create an anaconda environment with the yml file given here with:

1) conda env create -f environment.yml

2) conda activate kempner_ross_NDS2026_final_project

With vscode you can select the conda environment in the jupyter notebook.

In data/ you will find the the datasets for the two input-defined S1DZ 
subpopulations. Here you will find x,y coordinates of the mouse during the
social preference test and also the calcium traces.

In notebooks/ you will find the notebooks which produced all of the plots in
the slides I gave in class. The notebooks are organized in folders for the 
motor-S1DZ and sensory-S1DZ input-defined subpopulations.


**Welcome to the repo for Ross Kempner's NDS final project.**

**System requirements** 

This analysis has been tested only with an nvidia A6000 GPU with nvidia driver version 535.129.03, CUDA version 12.2, and OS Ubuntu 20.04.6 LTS.

I only have tested this with VSCode.

**Installation instructions** 

Create an anaconda environment with the environment.yml file given here with:

1) conda env create -f environment.yml

2) conda activate kempner_ross_NDS2026_final_project

With vscode you can select the conda environment in the jupyter notebook.

**How to recreate the figures in the paper**
In data/ you will find the the datasets for the two input-defined S1DZ 
subpopulations. Here you will find x,y coordinates of the mouse during the
social preference test and also the calcium traces.

In notebooks/ you will find the notebooks which produced all of the plots in
the slides I gave in class using this data.

The notebooks are organized in folders for the 
motor-S1DZ and sensory-S1DZ input-defined subpopulations.

Just run these notebooks with the conda environment in VScode jupyter notebook
and you will see the figures.