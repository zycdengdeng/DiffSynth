# Step 1: Initialize multi-control checkpoint from single-control weights
# python examples/wanvideo/model_training/scripts/Wan2.1-Fun-V1.1-14B-MultiControl-Initialize.py \
#   --input_path "models/PAI/Wan2.1-Fun-V1.1-14B-Control/diffusion_pytorch_model-00001-of-00007.safetensors,models/PAI/Wan2.1-Fun-V1.1-14B-Control/diffusion_pytorch_model-00002-of-00007.safetensors,..." \
#   --output_path "models/Wan2.1-Fun-V1.1-14B-MultiControl/diffusion_pytorch_model.safetensors"

# Step 2: Generate metadata.csv
# python examples/wanvideo/model_training/scripts/prepare_multi_control_metadata.py \
#   --data_root /mnt/zihanw/tf2.5_version1_full_Data

# Step 3: Train with three control inputs
DATA_ROOT=/mnt/zihanw/tf2.5_version1_full_Data

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_4gpu.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ${DATA_ROOT} \
  --dataset_metadata_path ${DATA_ROOT}/metadata.csv \
  --data_file_keys "video,control_video_sparse_depth,control_video_sparse_color,control_video_bbox" \
  --num_frames 29 \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths '["models/Wan2.1-Fun-V1.1-14B-MultiControl/diffusion_pytorch_model.safetensors"]' \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-MultiControl_full" \
  --trainable_models "dit" \
  --extra_inputs "control_video_sparse_depth,control_video_sparse_color,control_video_bbox" \
  --initialize_model_on_cpu
