#!/bin/bash
win=16          # trajectory length
img=1           # image sequence length
act=5           # action sequence length

lora_mode="lora" 
lora_rank=128
lora_alpha=32
lora_dropout=0.0
ttt_num_samples=34
ttt_traj_len=60
ttt_sample_repeat=6
ttt_batch_size=1
ttt_num_epoch=1
ttt_learning_rate=0.0005
ttt_weight_decay=0.01

ckpt_names=(
    "16" 
)

calvin_dataset_path="calvin/dataset/task_ABC_D/"
calvin_conf_path="calvin/calvin_models/conf"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
save_checkpoint_path="checkpoints/"
ttt_data_dir="./YOUR_DATA_DIR/"

window_size=$win
image_sequence_length=$img
state_sequence_length=$img  
action_sequence_length=$act
sequence_length=$(( (window_size - image_sequence_length) / action_sequence_length ))

experiment_name="scratch_qwen_${win}win_${img}img_${act}act"
log_name="scratch_qwen_${win}win_${img}img_${act}act_ttt_${ttt_num_samples}samples_${ttt_traj_len}traj_${ttt_sample_repeat}repeat_${lora_rank}rank_${lora_alpha}alpha_${ttt_learning_rate}lr_train_once"
base_checkpoint_dir="checkpoints/$experiment_name"

node=1
node_num=1

echo "========================================="
echo "Configuration:"
echo "Experiment name: $experiment_name"
echo "Window size: $window_size"
echo "Image sequence length: $image_sequence_length"
echo "State sequence length: $state_sequence_length"
echo "Action sequence length: $action_sequence_length"
echo "Sequence length: $sequence_length"
echo "Checkpoint directory: $base_checkpoint_dir"
echo "lora_mode: $lora_mode"
echo "lora_rank: $lora_rank" 
echo "lora_alpha: $lora_alpha"
echo "lora_dropout: $lora_dropout"
echo "ttt_num_samples: $ttt_num_samples"
echo "ttt_traj_len: $ttt_traj_len"
echo "ttt_sample_repeat: $ttt_sample_repeat"
echo "ttt_batch_size: $ttt_batch_size"
echo "ttt_num_epoch: $ttt_num_epoch"
echo "ttt_learning_rate: $ttt_learning_rate"
echo "ttt_weight_decay: $ttt_weight_decay"
echo "========================================="

for ckpt_name in "${ckpt_names[@]}"; do
    resume_from_checkpoint="${base_checkpoint_dir}/${ckpt_name}.pth"
    
    log_folder="eval_logs/$log_name"
    mkdir -p "$log_folder"
    log_file="eval_logs/$log_name/evaluate_${ckpt_name}.pth.log"

    torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10075 eval_calvin.py \
        --traj_cons \
        --rgb_pad -1 \
        --gripper_pad -1 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset ${calvin_dataset_path} \
        --calvin_conf_path ${calvin_conf_path} \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 1 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --num_obs_token_per_image 9 \
        --run_name ${experiment_name} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --transformer_layers 24 \
        --hidden_dim 384 \
        --transformer_heads 12 \
        --phase "evaluate" \
        --finetune_type "calvin" \
        --action_pred_steps 1 \
        --sequence_length ${sequence_length} \
        --future_steps 3 \
        --window_size ${window_size} \
        --obs_pred \
        --image_sequence_length ${image_sequence_length} \
        --state_sequence_length ${state_sequence_length} \
        --action_sequence_length ${action_sequence_length} \
        --use_qwen \
        --pred_image \
        --test_interval 1 \
        --ttt \
        --ttt_seed 42 \
        --ttt_num_samples ${ttt_num_samples} \
        --ttt_traj_len ${ttt_traj_len} \
        --ttt_sample_repeat ${ttt_sample_repeat} \
        --ttt_batch_size ${ttt_batch_size} \
        --ttt_num_epoch ${ttt_num_epoch} \
        --ttt_learning_rate ${ttt_learning_rate} \
        --ttt_weight_decay ${ttt_weight_decay} \
        --ttt_max_grad_norm 0.1 \
        --ttt_data_dir ${ttt_data_dir} \
        --lora_mode ${lora_mode} \
        --lora_rank ${lora_rank} \
        --lora_alpha ${lora_alpha} \
        --lora_dropout ${lora_dropout} \
        --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
done