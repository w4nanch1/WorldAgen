#!/bin/bash
export CUDA_HOME="/usr/local/cuda"
save_dir="./checkpoints/"
root_dir="./datasets"
vit_checkpoint_path="./checkpoints/mae_pretrain_vit_base.pth"
libero_path="./datasets/libero_90"

calvin_dataset_path="calvin/dataset/task_ABC_D"
node=1
node_num=4
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10711 train_libero.py \
    --traj_cons \
    --rgb_pad -1 \
    --gripper_pad -1 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 48 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_traj7_len_1img_3act \
    --save_checkpoint_path ${save_dir} \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 1 \
    --sequence_length 2 \
    --image_sequence_length 1 \
    --state_sequence_length 1 \
    --action_sequence_length 3 \
    --future_steps 3 \
    --window_size 7 \
    --loss_image \
    --loss_action \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 1 \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path ${libero_path} \
    --report_to_wandb \
    --use_qwen \
    --pred_image \