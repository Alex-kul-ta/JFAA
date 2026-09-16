#!/bin/bash -l
#SBATCH --job-name=jfaa_vjepa2_1-vitG-384-mindwell-full-epoch
#SBATCH --clusters=mindwell
#SBATCH --account=lp_gevam
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_b200
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexandre.personnic@kuleuven.be
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

conda activate sta

cd /data/leuven/367/vsc36743/codebase/JFAA/

output_dir="${VSC_SCRATCH_NODE}/experiments/JFAA/ek100_vjepa2_1_vitG_384_$(date +%Y%m%d_%H%M%S)"

mkdir -p $output_dir

python -m evals.main --fname configs/jfaa_vjepa2_vitG_train_mindwell.yaml --devices cuda:0 --master_addr 127.0.0.1 --master_port auto --folder $output_dir --override_config_folder

irsync -r ${VSC_SCRATCH_NODE}/experiments i:/set/home/ciis-lab/experiments