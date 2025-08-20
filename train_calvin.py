import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.model import UniAorld
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.calvin_data_utils import get_calvin_dataset, get_calvin_val_dataset, get_calvin_test_dataset, get_libero_pretrain_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  
from accelerate import Accelerator
from accelerate.utils import set_seed

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params

@record
def main(args):
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("Seer", f"Seer_ptbs{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}")
    print("run_name:", args.run_name)
    # print('pred_state', args.pred_state)
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
        action_pred_steps=args.action_pred_steps,
        pred_image=args.pred_image,
        atten_goal=args.atten_goal,
        use_qwen=args.use_qwen
    )
    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "libero_pretrain":
        calvin_dataset = get_libero_pretrain_dataset(args, model.image_processor, clip, epoch=0)
    # calvin_dataset = get_calvin_test_dataset(args, model.image_processor, clip, epoch=0)
    calvin_loader = calvin_dataset.dataloader
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
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
    total_params, trainable_params = count_parameters(model)
    print("total_params: {} M".format(total_params/1024/1024))
    print("trainable_params: {} M".format(trainable_params/1024/1024))
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)  # TODO make sure the parameters which need to be optimized are passing
    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    resume_from_epoch = 0
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")    
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")
        image_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "image_decoder" in k]
        projector_keys = [k for k in checkpoint["model_state_dict"].keys() if "projector" in k]
        action_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "action_decoder" in k]
        if args.reset_action_token:
            del checkpoint["model_state_dict"]["module.action_pred_token"] 
        if args.reset_obs_token:
            del checkpoint["model_state_dict"]["module.obs_tokens"] 
        if args.reset_mask_token:
            del checkpoint["model_state_dict"]["module.mask_token"] 
        if args.reset_image_decoder:
            for k in image_decoder_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
        if args.reset_action_decoder:
            for k in action_decoder_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
        if checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"].shape != ddp_model.module.transformer_backbone_position_embedding.shape:
            checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"] = checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"][:, :args.sequence_length, :, :]
        print("loading pretrained weights :", checkpoint["model_state_dict"].keys())
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {resume_from_epoch}")
 
    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ## accelerete 
    # if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
    #     accelerator = Accelerator(mixed_precision='bf16')
    # else:
    #     accelerator = Accelerator(mixed_precision='fp16')
    # model, optimizer,lr_scheduler, calvin_loader = accelerator.prepare(
    #     model, optimizer,lr_scheduler, calvin_loader)


    ddp_model.train()
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        
        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
            use_diffusion_head=False,
            image_processor=model.image_processor,
        )
        if args.rank == 0 and args.save_checkpoint and epoch % args.save_checkpoint_seq == 0 and epoch > args.start_save_checkpoint:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    