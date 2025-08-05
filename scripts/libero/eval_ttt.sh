export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export EGL_PLATFORM="surfaceless"
export LIBGL_ALWAYS_SOFTWARE="1"

lora_mode="lora"
lora_rank=64
lora_alpha=32
lora_dropout=0.0
ttt_num_samples=5
ttt_traj_len=60
ttt_sample_repeat=6
ttt_batch_size=1
ttt_num_epoch=1
ttt_learning_rate=0.0005
ttt_weight_decay=0.01
ttt_data_dir="./TTT_DATA_DIR/"

pthlist=("38")
for ckpt_id in "${pthlist[@]}"; do
    resume_from_checkpoint="./YOUR_CHECKPOINT_DIR/"
    vit_checkpoint_path="./checkpoints/mae_pretrain_vit_base.pth"
    this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
    save_checkpoint_path="./YOUR_CHECKPOINT_SAVE_DIR/"
    dirname=$(basename "$resume_from_checkpoint")
    LOG_DIR="./YOUR_LOG_DIR/"
    mkdir -p ${LOG_DIR}
    test_id="${ckpt_id}_ttt_${ttt_sample_repeat}repeat_${lora_rank}rank_${ttt_learning_rate}lr_${ttt_num_samples}samples_${ttt_sample_repeat}repeat"
    logfile="${LOG_DIR}/${test_id}.log"
    run_name="${dirname}_${test_id}"

    node=1
    node_num=1
 
    torchrun  --nnodes=${node} --nproc_per_node=${node_num} --master_port=10022 eval_libero.py \
        --traj_cons \
        --rgb_pad -1 \
        --gripper_pad -1 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset "" \
        --workers 8 \
        --lr_scheduler cosine \
        --save_every_iter 100000 \
        --num_epochs 40 \
        --seed 42 \
        --batch_size 256 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name ${run_name} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --transformer_layers 24 \
        --phase "evaluate" \
        --finetune_type "libero_10" \
        --action_pred_steps 1 \
        --sequence_length 2 \
        --image_sequence_length 1 \
        --state_sequence_length 1 \
        --action_sequence_length 3 \
        --pred_image \
        --future_steps 3 \
        --window_size 7 \
        --obs_pred \
        --use_qwen \
        --gripper_width \
        --eval_libero_ensembling \
        --ttt \
        --ttt_seed 42 \
        --ttt_num_samples ${ttt_num_samples} \
        --ttt_use_test_text_instruction \
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
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}
done