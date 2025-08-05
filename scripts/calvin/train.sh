#!/bin/bash
calvin_dataset_path="calvin/dataset/task_ABC_D"
save_checkpoint_path="checkpoints/"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"

node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train_calvin.py \
    --traj_cons \
    --rgb_pad -1 \
    --gripper_pad -1 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 2 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --finetune_type "calvin" \
    --wandb_project YOUR_WANDB_PROJECT_NAME \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name test \
    --save_checkpoint \
    --start_save_checkpoint 1 \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 1 \
    --sequence_length 5 \
    --future_steps 3 \
    --window_size 16 \
    --pred_image \
    --loss_image \
    --loss_action \
    --obs_pred \
    --use_qwen \
    --image_sequence_length 1 \
    --state_sequence_length 1 \
    --action_sequence_length 3 \
    --warmup_epochs 5 \
    --report_to_wandb \
    