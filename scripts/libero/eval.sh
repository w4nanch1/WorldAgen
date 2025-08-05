export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export EGL_PLATFORM="surfaceless"
export LIBGL_ALWAYS_SOFTWARE="1"
pthlist=("38")
for ckpt_id in "${pthlist[@]}"; do
    resume_from_checkpoint="/workspace/root/uniaorld/scripts/LIBERO_LONG/Seer/checkpoints/libero_scratch_seq7_len_1img_3act"
    vit_checkpoint_path="/workspace/root/uniaorld/checkpoints/mae_pretrain_vit_base.pth"
    this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
    save_checkpoint_path="/workspace/root/uniaorld/scripts/LIBERO_LONG/Seer/checkpoints/libero_scratch_seq10_len/libero_scratch"
    dirname=$(basename "$resume_from_checkpoint")
    LOG_DIR="/workspace/root/uniaorld/scripts/LIBERO_LONG/Seer/log/${dirname}"
    mkdir -p ${LOG_DIR}
    test_id="${ckpt_id}_ori"
    logfile="${LOG_DIR}/${test_id}.log"
    run_name="${dirname}_${test_id}"

    node=1
    node_num=1
 
    torchrun  --nnodes=${node} --nproc_per_node=${node_num} --master_port=10012 eval_libero.py \
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
        --future_steps 3 \
        --window_size 7 \
        --obs_pred \
        --use_qwen \
        --test_ori \
        --pred_image \
        --gripper_width \
        --eval_libero_ensembling \
        --eval_action_entropy \
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}
done