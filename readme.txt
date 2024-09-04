Installation 
This package is compatible with Python 3.8.

1. Installing PyTorch
First, please install PyTorch version 1.12.1 following the official instructions. Ensure your environment supports CUDA 11.6. You can use the following command for installation:

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

Note: For GPU support, it is crucial to install specific versions of CUDA and cuDNN that are compatible with the respective version of PyTorch.

2. Installing Dependencies
ApuShape relies on the following excellent packages:

matplotlib
scipy
monai
scikit-image
csbdeep
numba
stardist

These dependencies will be automatically installed via conda or pip if they are not already present in your environment. You can ensure all dependencies are met by running:

pip install -r requirements.txt

Usage
We provide example workflow for ApuShape and ApuRefine

main.py contains the complete ApuShape workflow.
refine.py runs ApuRefine on top of Mesmer results.

To configure the workflows, you only need to modify the parameters in the params.ini file.
