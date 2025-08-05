import os
import random
import numpy as np
import torch
import wandb

import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from models.model import UniAorld
from utils.calvin_data_utils import get_calvin_dataset, get_calvin_val_dataset, get_calvin_test_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.eval_utils_calvin import test_calvin
from utils.arguments_utils import get_parser
from test_time_training.ttt import TestTimeTrainer

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)

    model = UniAorld(
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        image_sequence_length=args.image_sequence_length,
        state_sequence_length=args.state_sequence_length,
        action_sequence_length=args.action_sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        pred_state=args.pred_state,
        pred_image=args.pred_image,
        atten_goal=args.atten_goal,
        use_qwen=args.use_qwen,
        action_pred_steps=args.action_pred_steps,
        eval_action_entropy=args.eval_action_entropy,
    )
    random_seed(args.seed, args.rank)
    print(f"Start running evaluation on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module:
            model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    ddp_model.eval()
    eval_log_dir = 'evaluate'
    
    if args.finetune_type == "calvin":
        test_calvin( 
            args=args,
            model=ddp_model,
            image_processor=model.image_processor,
            tokenizer=clip,
            dataset_path=args.calvin_dataset,
            eval_log_dir=eval_log_dir,
            debug=args.visualize,
            reset=args.reset,
            diverse_inst=args.diverse_inst,
            ttt=args.ttt,
        )
    else:
        raise NotImplementedError        

if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()