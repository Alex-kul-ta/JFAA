#!/bin/bash -l
#SBATCH --job-name=jfaa_vjepa2_1-vitG-384-mindwell
#SBATCH --clusters=mindwell
#SBATCH --account=lp_gevam
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_b200
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexandre.personnic@kuleuven.be
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

conda activate sta

cd /data/leuven/367/vsc36743/codebase/JFAA/

python -m evals.main --fname configs/jfaa_vjepa2_vitG_train_mindwell.yaml --devices cuda:0 --master_addr 127.0.0.1 --master_port auto