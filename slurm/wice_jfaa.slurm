#!/bin/bash -l
#SBATCH --job-name=jfaa_vjepa2_1-vitG-384-wice
#SBATCH --clusters=wice
#SBATCH --account=lp_gevam
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=18
#SBATCH --partition=gpu_a100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexandre.personnic@kuleuven.be
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Running on host $(hostname)"
echo "================================"

mkdir -p $VSC_SCRATCH_NODE/EPIC-KITCHENS

icd /set/home/ciis-lab/datasets/

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Syncing EPIC-KITCHENS dataset from Mango to $VSC_SCRATCH_NODE/EPIC-KITCHENS"
echo "================================"

irsync -r i:epic-kitchens $VSC_SCRATCH_NODE/EPIC-KITCHENS

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Untarring EPIC-KITCHENS frames to $VSC_SCRATCH_NODE/EPIC-KITCHENS/rgb_frames"
echo "================================"

python /gpfs1/scratch/367/vsc36743/EPIC-KITCHENS/utils_ek100_dataset.py --rgb_frames_dir $VSC_SCRATCH_NODE/EPIC-KITCHENS/rgb_frames/ --mode untar --rm --k 70

output_dir="${VSC_SCRATCH_NODE}/experiments/JFAA/ek100_vjepa2_1_vitG_384_$(date +%Y%m%d_%H%M%S)"

mkdir -p $output_dir

cat > $output_dir/jfaa_vjepa2_1_vitG_384_gaze.yaml <<EOF
nodes: 1
tasks_per_node: 1
cpus_per_task: 72

tag: ek100-vjepa2_1-vitG-384-wice
eval_name: action_anticipation_frozen
folder: $output_dir
resume_checkpoint: true

wandb:
  enabled: true
  project: JFAA
  watch: false

experiment:
  classifier:
    num_probe_blocks: 4
    num_heads: 16

  data:
    dataset: EK100
    input_format: frames
    file_format: 0

    base_path: ${VSC_SCRATCH_NODE}/EPIC-KITCHENS/rgb_frames
    dataset_train: ${VSC_SCRATCH_NODE}/EPIC-KITCHENS/annotations/EPIC_100_train.csv
    dataset_val: ${VSC_SCRATCH_NODE}/EPIC-KITCHENS/annotations/EPIC_100_validation.csv
    video_info_path: ${VSC_SCRATCH_NODE}/EPIC-KITCHENS/annotations/EPIC_100_video_info.csv

    frame_template: "frame_{:010d}.jpg"
    frame_index_offset: 1

    anticipation_time_sec:
      - 1.0
      - 1.0

    auto_augment: true
    frames_per_clip: 32
    frames_per_second: 8
    motion_shift: false
    num_workers: 16
    pin_memory: true
    random_resize_scale:
      - 0.08
      - 1.0
    reprob: 0.25
    resolution: 384

    # In the sampler formula, point=1.0 means start_frame - anticipation_time.
    # This matches the report protocol: no frames after t_s - T_a are used.
    train_anticipation_point:
      - 1.0
      - 1.0
    train_anticipation_time_sec:
      - 0.25
      - 1.75
    val_anticipation_point:
      - 1.0
      - 1.0

  optimization:
    num_epochs: 6
    batch_size: 6
    use_bfloat16: true
    use_focal_loss: true

    multihead_kwargs:
      - {weight_decay: 0.0001, final_weight_decay: 0.0001, lr: 0.005, start_lr: 0.005, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.0001, final_weight_decay: 0.0001, lr: 0.003, start_lr: 0.003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.0001, final_weight_decay: 0.0001, lr: 0.001, start_lr: 0.001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.0001, final_weight_decay: 0.0001, lr: 0.0003, start_lr: 0.0003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.0001, final_weight_decay: 0.0001, lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.001, final_weight_decay: 0.001, lr: 0.005, start_lr: 0.005, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.001, final_weight_decay: 0.001, lr: 0.003, start_lr: 0.003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.001, final_weight_decay: 0.001, lr: 0.001, start_lr: 0.001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.001, final_weight_decay: 0.001, lr: 0.0003, start_lr: 0.0003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.001, final_weight_decay: 0.001, lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.01, final_weight_decay: 0.01, lr: 0.005, start_lr: 0.005, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.01, final_weight_decay: 0.01, lr: 0.003, start_lr: 0.003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.01, final_weight_decay: 0.01, lr: 0.001, start_lr: 0.001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.01, final_weight_decay: 0.01, lr: 0.0003, start_lr: 0.0003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.01, final_weight_decay: 0.01, lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.1, final_weight_decay: 0.1, lr: 0.005, start_lr: 0.005, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.1, final_weight_decay: 0.1, lr: 0.003, start_lr: 0.003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.1, final_weight_decay: 0.1, lr: 0.001, start_lr: 0.001, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.1, final_weight_decay: 0.1, lr: 0.0003, start_lr: 0.0003, final_lr: 0.0, warmup: 0.0}
      - {weight_decay: 0.1, final_weight_decay: 0.1, lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0}

model_kwargs:
  checkpoint: /gpfs1/scratch/367/vsc36743/vjepa2_checkpoints/vjepa21_v2_lr_0.0001_wd_0.0005_lora_r_64.pt
  module_name: evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar

  wrapper_kwargs:
    no_predictor: false
    num_output_frames: 2
    num_steps: 1

  pretrain_kwargs:
    use_v2_1: true
    custom: true
    encoder:
      model_name: vit_giant_xformers
      checkpoint_key: target_encoder
      tubelet_size: 2
      patch_size: 16
      uniform_power: false
      use_rope: true
      use_sdpa: true
      use_silu: false
      wide_silu: true
      img_temporal_dim_size: 1
      interpolate_rope: true
    predictor:
      model_name: vit_predictor
      checkpoint_key: predictor
      num_frames: 64
      depth: 24
      num_heads: 12
      predictor_embed_dim: 384
      num_mask_tokens: 8
      uniform_power: false
      use_mask_tokens: true
      use_sdpa: true
      use_silu: false
      wide_silu: true
      use_rope: true
      img_temporal_dim_size: 1
      n_output_distillation: 4
      return_all_tokens: true
EOF

cd /data/leuven/367/vsc36743/codebase/JFAA/

conda activate sta

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Starting training with config: $output_dir/jfaa_vjepa2_1_vitG_384_gaze.yaml"
echo "Saving outputs to: $output_dir"
echo "================================"

python -m evals.main --fname $output_dir/jfaa_vjepa2_1_vitG_384_gaze.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 --master_addr 127.0.0.1 --master_port auto

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Syncing experiments from $VSC_SCRATCH_NODE/experiments to Mango"
echo "================================"

irsync -r ${VSC_SCRATCH_NODE}/experiments i:/set/home/ciis-lab/experiments

echo "================================"
echo "$(date +'%Y-%m-%d %H:%M:%S'): Job completed"
echo "================================"