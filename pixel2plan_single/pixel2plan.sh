#!/bin/bash

#SBATCH -n 8
#SBATCH -c 1
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -t 24:00:00
#SBATCH -e /home/lutter/projects/pixel2plan/cluster_logs/%A_pixel2plan.err.
#SBATCH -o /home/lutter/projects/pixel2plan/cluster_logs/%A_pixel2plan.out.

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/lutter/open_source/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/lutter/open_source/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lutter/open_source/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/lutter/open_source/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate pixel2plan
mpiexec -np 8 python -u main_mpi.py
