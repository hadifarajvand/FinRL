#!/bin/bash

echo "Starting Jupyter Notebook with FinRL DRL Environment..."
echo "Kernel: FinRL DRL Environment (drl_env_new)"
echo ""
 
# Activate the virtual environment and start Jupyter
/home/ubuntu/.pyenv/versions/drl_env_new/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root 