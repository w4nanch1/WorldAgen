import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_droid(
    args,
    model,
    epoch,
    droid_loader,
    optimizer,
    lr_scheduler,
    device_id,
):
    num_batches_per_epoch_droid = droid_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_droid
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(droid_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch in t:
        data_time_m.update(time.time() - end)

        # images
        images_primary_1 = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_primary_2 = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch[2].to(device_id, dtype=cast_dtype, non_blocking=True)

        # text tokens
        text_tokens = batch[5].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        
        # states
        states = batch[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2 
        
        # actions
        actions = batch[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        # actions[..., 6:] = (actions[..., 6:] + 1) // 2
        input_image_primary_1 = images_primary_1[:, :args.sequence_length, :]
        input_image_primary_2 = images_primary_2[:, :args.sequence_length, :]

        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]
        input_action = actions[:, :args.sequence_length, :]
        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 

        with autocast():  # image_primary, image_wrist, state, language_instruction
            arm_pred_action, gripper_pred_action, image_pred = model(
                image_primary_1=input_image_primary_1,
                image_primary_2=input_image_primary_2,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=input_action,
            )
        # loss_action
        if args.loss_action and args.action_pred_steps:
            loss_arm_action = torch.nn.functional.smooth_l1_loss(
                            arm_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, :6].detach())
            loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                            gripper_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, 6:].detach())
        else:
            loss_arm_action = torch.tensor([0.0]).to(device_id)
            loss_gripper_action = torch.tensor([0.0]).to(device_id)

        # loss_image 
        if args.loss_image and args.obs_pred:
            label_image_primary_1 = images_primary_1[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_primary_2 = images_primary_2[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_wrist = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)

            label_image_primary_1 = patchify(label_image_primary_1, patch_size=args.patch_size)
            label_image_primary_2 = patchify(label_image_primary_2, patch_size=args.patch_size)
            label_image_wrist = patchify(label_image_wrist, patch_size=args.patch_size)

            label_image_primary_1 = normalize_patchfied_image(label_image_primary_1)
            label_image_primary_2 = normalize_patchfied_image(label_image_primary_2)
            label_image_wrist = normalize_patchfied_image(label_image_wrist)

            image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
            image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            loss_image = (torch.nn.functional.mse_loss(
                            image_pred[:, 0, :, :], 
                            label_image_primary_1.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 1, :, :], 
                            label_image_primary_2.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 2, :, :], 
                            label_image_wrist.detach())) / 3
        else:
            loss_image = torch.tensor([0.0]).to(device_id)
        loss_droid = args.loss_arm_action_ratio * loss_arm_action + args.loss_gripper_action_ratio * loss_gripper_action + 0.1 * loss_image

        # gradient_accumulation_steps        
        loss = loss_droid / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_droid.item(), "loss_image": loss_image.item(), "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item()})

        # if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
                
        #     if args.rank == 0:
        #         import os
        #         if not os.path.exists(f"{args.save_checkpoint_path}/exp/{args.run_name}"):
        #             os.makedirs(f"{args.save_checkpoint_path}/exp/{args.run_name}")

        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }

        #         ckpt_name = get_ckpt_name(args, global_step)
        #         ckpt_path = os.path.join(f"{args.save_checkpoint_path}/exp", args.run_name, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}")
        #         torch.save(checkpoint_dict, ckpt_path)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(ckpt_path)

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    use_diffusion_head=False,
    image_processor=None,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # images
        images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        # text tokens
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)

        # states
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2

        num_chunk = args.sequence_length
        images_primary = images_primary[:, :args.window_size - (args.action_pred_steps - 1) * args.action_sequence_length, ...]
        images_wrist = images_wrist[:, :args.window_size - (args.action_pred_steps - 1) * args.action_sequence_length, ...]
        input_states = input_states[:, :args.window_size - (args.action_pred_steps - 1) * args.action_sequence_length, ...]

        # actions
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        actions[..., 6:] = (actions[..., 6:] + 1) // 2

        assert (args.window_size - args.image_sequence_length) % args.action_sequence_length == 0
        assert args.image_sequence_length == args.state_sequence_length

        input_image_primary, label_image_primary = sample(images_primary, args, num_chunk)
        input_image_wrist, label_image_wrist = sample(images_wrist, args, num_chunk)
        input_state, label_state = sample(input_states, args, num_chunk)
        input_text_token = text_tokens[:, :num_chunk, ...]

        input_action = actions[:, args.image_sequence_length : args.window_size - (args.action_pred_steps - 1) * args.action_sequence_length, ...]
        label_action = torch.cat([actions[:, args.image_sequence_length + j * args.action_sequence_length : \
                                          args.window_size - (args.action_pred_steps - 1) * args.action_sequence_length + j * args.action_sequence_length, ...].unsqueeze(-2) \
                                            for j in range(args.action_pred_steps)], dim=-2)
        label_action = label_action.view(label_action.shape[0], num_chunk, -1, label_action.shape[-1])
        label_state = label_state.view(label_state.shape[0], num_chunk, -1, label_state.shape[-1])
        
        # print(input_image_primary.shape) # torch.Size([2, 16, 3, 224, 224])
        # visualize_video_as_gif(input_image_primary)
        # time.sleep(9992)
        if use_diffusion_head:
            label_image_primary = patchify(label_image_primary.flatten(0, 1), patch_size=args.patch_size)
            label_image_wrist = patchify(label_image_wrist.flatten(0, 1), patch_size=args.patch_size)

            # label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 
            with autocast():  # image_primary, image_wrist, state, language_instruction
                loss_primary, loss_wrist, loss_state, loss_action = model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=input_action,
                    label_action=label_action,
                    label_wrist=label_image_wrist,
                    label_primary=label_image_primary, 
                    label_state=label_state,
                )
            loss_image = (loss_primary + loss_wrist) * 0.5
            loss_calvin = loss_action + loss_state + 0.1 * loss_image

            loss = loss_calvin / args.gradient_accumulation_steps
            loss_action = loss_action / args.gradient_accumulation_steps
            loss_state = loss_state / args.gradient_accumulation_steps
            loss_image = loss_image / args.gradient_accumulation_steps
            mv_avg_loss.append(loss.item())

            ### backward pass ###
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # step optimizer and log
            if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
                num_steps == num_batches_per_epoch - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # step time and reset end outside of rank 0
                step_time_m.update(time.time() - end)
                end = time.time()

                if args.rank == 0 and args.report_to_wandb:
                    calvin_samples_per_second = (
                        args.gradient_accumulation_steps
                        * args.batch_size
                        * args.world_size
                        / step_time_m.val
                    )
                    calvin_samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps
                        * args.batch_size
                        / step_time_m.val
                    )

                    wandb.log(
                        {
                            "data_time": data_time_m.avg,
                            "step_time": step_time_m.avg,
                            "calvin_samples_per_second": calvin_samples_per_second,
                            "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )
                    step_time_m.reset()
                    data_time_m.reset()

                    wandb.log(
                        {
                            "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                            "loss_action": loss_action.item() * args.gradient_accumulation_steps,
                            "loss_state": loss_state.item() * args.gradient_accumulation_steps,
                            "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                            "global_step": global_step,
                        },
                    )

            avg_horizon = min(100, len(mv_avg_loss))
            t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "loss_image": loss_image.item(), "loss_action": loss_action.item(), "loss_state": loss_state.item()})
        
        else:
            with autocast():  # image_primary, image_wrist, state, language_instruction
                arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state = model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=input_action,
                )
            # loss_action
            if args.loss_action:
                loss_arm_action = torch.nn.functional.smooth_l1_loss(
                                arm_pred_action, 
                                label_action[..., :6].detach())
                loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                                gripper_pred_action, 
                                label_action[..., 6:].detach())
            else:
                loss_arm_action = torch.tensor([0.0]).to(device_id)
                loss_gripper_action = torch.tensor([0.0]).to(device_id)
            
            if args.loss_state and args.pred_state:
                loss_arm_state = torch.nn.functional.smooth_l1_loss(
                                arm_pred_state, 
                                label_state[..., :6].detach())
                loss_gripper_state = torch.nn.functional.binary_cross_entropy(
                                gripper_pred_state, 
                                label_state[..., 6:].detach())
            else:
                loss_arm_state = torch.tensor([0.0]).to(device_id)
                loss_gripper_state = torch.tensor([0.0]).to(device_id)

            # loss_image 
            if args.loss_image and args.pred_image:
                label_image_primary = patchify(label_image_primary.flatten(0, 1), patch_size=args.patch_size)
                label_image_wrist = patchify(label_image_wrist.flatten(0, 1), patch_size=args.patch_size)

                # label_image_primary = normalize_patchfied_image(label_image_primary)
                # label_image_wrist = normalize_patchfied_image(label_image_wrist)
                loss_image = 0.5 * (torch.nn.functional.mse_loss(
                                image_pred[:, 0, :, :], 
                                label_image_primary.detach()) + 
                                torch.nn.functional.mse_loss(
                                image_pred[:, 1, :, :], 
                                label_image_wrist.detach()))
            else:
                loss_image = torch.tensor([0.0]).to(device_id)
            loss_calvin = args.loss_arm_action_ratio * loss_arm_action + args.loss_gripper_action_ratio * loss_gripper_action \
                + 0.1 * loss_image + 0.01 * loss_arm_state + 0.01 * loss_gripper_state
            # gradient_accumulation_steps
            loss = loss_calvin / args.gradient_accumulation_steps
            loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
            loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
            loss_arm_state = loss_arm_state / args.gradient_accumulation_steps
            loss_gripper_state = loss_gripper_state / args.gradient_accumulation_steps
            loss_image = loss_image / args.gradient_accumulation_steps

            mv_avg_loss.append(loss.item())

            ### backward pass ###
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # step optimizer and log
            if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
                num_steps == num_batches_per_epoch - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # step time and reset end outside of rank 0
                step_time_m.update(time.time() - end)
                end = time.time()

                if args.rank == 0 and args.report_to_wandb:
                    calvin_samples_per_second = (
                        args.gradient_accumulation_steps
                        * args.batch_size
                        * args.world_size
                        / step_time_m.val
                    )
                    calvin_samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps
                        * args.batch_size
                        / step_time_m.val
                    )

                    wandb.log(
                        {
                            "data_time": data_time_m.avg,
                            "step_time": step_time_m.avg,
                            "calvin_samples_per_second": calvin_samples_per_second,
                            "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )
                    step_time_m.reset()
                    data_time_m.reset()

                    log_dict = {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_arm_action": loss_arm_action.item() * args.gradient_accumulation_steps,
                        "loss_gripper_action": loss_gripper_action.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    }
                    if args.loss_image and args.pred_image:
                        log_dict["loss_image"] = loss_image.item() * args.gradient_accumulation_steps
                    if args.pred_state and args.loss_state:
                        log_dict["loss_arm_state"]   = loss_arm_state.item() * args.gradient_accumulation_steps
                        log_dict["loss_gripper_state"] = loss_gripper_state.item() * args.gradient_accumulation_steps

                    wandb.log(log_dict)

            avg_horizon = min(100, len(mv_avg_loss))
            tqdm_dict = {"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), \
                           "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item()} 
            if args.pred_state and args.loss_state:
                tqdm_dict["loss_arm_state"] = loss_arm_state.item()
                tqdm_dict["loss_gripper_state"] = loss_gripper_state.item()
            if args.pred_image and args.loss_image:
                tqdm_dict[ "loss_image"] = loss_image.item()
            t.set_postfix(tqdm_dict)

def get_robot_info_from_multi_step_env(env, actions, image_process_fn):
    B, action_seq_len, _ = actions.shape
    primary_list = []
    wrist_list = []
    state_list = []
    current_info_list = []
    from PIL import Image
    for i in range(action_seq_len):
        action = actions[0, i, ...].cpu().detach().to(dtype=torch.float16).numpy()
        action[-1] = np.where(action[-1] >= 0, 1.0, -1.0)
        obs, _, _, curren_info = env.step(action)
        state_list.append(torch.from_numpy(obs["robot_obs"]).unsqueeze(0).unsqueeze(1))
        primary_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_static"])]).unsqueeze(0))
        wrist_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_gripper"])]).unsqueeze(0))
        current_info_list.append(curren_info)
    primary = torch.cat(primary_list, dim=1)
    wrist = torch.cat(wrist_list, dim=1)
    states = torch.cat(state_list, dim=1)
    return primary, wrist, states, current_info_list

def sample(input_tensor, args, num_chunk):
    input_list = []
    label_list = []
    input_list.append(input_tensor[:, :args.image_sequence_length, ...])
    for i in range(num_chunk):
        block = input_tensor[:, args.image_sequence_length+i*args.action_sequence_length:args.image_sequence_length+(i+1)*args.action_sequence_length, ...]
        if args.image_sequence_length == 1:
            indices = [args.action_sequence_length - 1]
        else:
        # indices = [(j * args.action_sequence_length) // args.image_sequence_length for j in range(args.image_sequence_length)]
            indices = torch.linspace(0, args.action_sequence_length-1, steps=args.image_sequence_length) \
                         .round() \
                         .long() \
                         .tolist()
        samples = [block[:, idx, ...].unsqueeze(1) for idx in indices]
        samples = torch.cat(samples, dim=1)
        if i != num_chunk - 1:
            input_list.append(samples)
        label_list.append(samples)
    input_list = torch.cat(input_list, dim=1)
    label_list = torch.cat(label_list, dim=1)
    return input_list, label_list

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visualize_video_as_gif(video_tensor, batch_idx=0, save_path="video.gif", fps=5):
    """
    Visualize a video tensor as a GIF.
    
    Args:
        video_tensor (torch.Tensor): Tensor with shape (batch_size, frames, channels, height, width)
        batch_idx (int): Which batch to visualize (default: 0)
        save_path (str): Path to save the GIF (default: "video.gif")
        fps (int): Frames per second for the GIF (default: 5)
    """
    # Check if the tensor has the expected shape
    assert len(video_tensor.shape) == 5, "Expected a 5D tensor (batch, frames, channels, height, width)"
    
    # Extract the specific batch
    video = video_tensor[batch_idx]
    
    # Convert to numpy and ensure it's in the right format for imageio
    # (frames, height, width, channels) with values between 0 and 255
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    
    # Transpose from (frames, channels, height, width) to (frames, height, width, channels)
    video = np.transpose(video, (0, 2, 3, 1))
    
    # # Normalize to [0, 1] if needed
    # if video.max() > 1.0:
    #     video = video / 255.0
    
    # Ensure values are between 0 and 1
    video = np.clip(video, 0, 1)
    
    # Convert to uint8 for imageio
    video_uint8 = (video * 255).astype(np.uint8)
    
    # Save as GIF
    import imageio
    imageio.mimsave(save_path, video_uint8, fps=fps)
    
    print(f"GIF saved to {save_path}")
    
    return save_path
