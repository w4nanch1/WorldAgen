import torch
import torch.nn as nn
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import functools
import json 
from torch.utils.data import Dataset, DataLoader
from test_time_training.lora import LoraModel
import torch.optim as optim
from tqdm.auto import tqdm, trange
import PIL.Image as Image
import random
from UniAorld.utils.train_utils import patchify, normalize_patchfied_image
import os 
import numpy as np
import random
from pathlib import Path
from omegaconf import OmegaConf
import math
import time
import shutil
from utils.calvin_data_utils import preprocess_image, preprocess_text_calvin
from transformers import get_cosine_schedule_with_warmup
from utils.train_utils import get_cast_dtype, get_autocast

# TRAJ_LEN = 360
RANDOM_CHOOSE_REPEAT = 1
class RandomSampler():
    def __init__(self, 
                 wrapped_model, 
                 env, 
                 args, 
                 eval_seq_path, 
                 lang_path, 
                 text_process_fn,
                 image_process_fn,
                 test_idx):
        set_seed(args.seed)
        self.test_idx = test_idx
        self.wrapped_model = wrapped_model
        self.args = args
        self.env = env
        self.eval_seq_path = eval_seq_path
        self.lang_path = lang_path
        self.text_process_fn = text_process_fn
        self.image_process_fn = image_process_fn
        self.num_chunk = args.sequence_length

    def reset_env(self, initial_state):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.wrapped_model.reset()

    def sample_one_traj(self, val_annotations, subtask, save_dir, traj_idx):
        obs = self.env.get_obs()

        lang_annotation = val_annotations[subtask][0]
        lang_annotation = lang_annotation.split('\n')[0]
        if '\u2019' in lang_annotation:
            lang_annotation.replace('\u2019', '\'')
        
        text = self.text_process_fn([lang_annotation])
        no_text = "no lang"
        no_text = self.text_process_fn([no_text])
        
        image_primary = obs["rgb_obs"]['rgb_static']
        image_primary = Image.fromarray(image_primary)
        image_primary = self.image_process_fn([image_primary])
        image_primary = torch.cat([image_primary.unsqueeze(1) for _ in range(self.args.image_sequence_length)], dim=1)

        image_wrist = obs["rgb_obs"]['rgb_gripper']
        image_wrist = Image.fromarray(image_wrist)
        image_wrist = self.image_process_fn([image_wrist])
        image_wrist = torch.cat([image_wrist.unsqueeze(1) for _ in range(self.args.image_sequence_length)], dim=1)

        state = obs['robot_obs']
        state = torch.from_numpy(np.stack([state]))
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)
        state = torch.cat([state.unsqueeze(1) for _ in range(self.args.state_sequence_length)], dim=1)

        sub_actions = []
        sub_primary = []
        sub_wrist = []
        sub_state = []

        for i in range(self.args.ttt_traj_len):
            if i == 0:
                first_action = torch.zeros(1, self.args.action_sequence_length, 7)
                if self.args.eval_action_entropy:
                    action_pred, _ = self.wrapped_model.step_agent(text, state, image_primary, image_wrist, action=first_action)
                else:
                    action_pred = self.wrapped_model.step_agent(text, state, image_primary, image_wrist, action=first_action) # [B, action_seq, 7]
            else:
                if self.args.eval_action_entropy:
                    action_pred, _ = self.wrapped_model.step_agent(text, state, image_primary, image_wrist)
                else:
                    action_pred = self.wrapped_model.step_agent(text, state, image_primary, image_wrist)
            _, _, _ = self.wrapped_model.step_world_model(action_pred)
            # action_pred = self.add_noise_to_action(action_pred) # TODO
            sub_actions.append(action_pred)
            image_primary, image_wrist, state, _ = self.get_robot_info_from_multi_step_env(action_pred, self.image_process_fn)
            sub_primary.append(image_primary)
            sub_wrist.append(image_wrist)
            sub_state.append(state)
            
            indices = [(j * self.args.action_sequence_length) // self.args.image_sequence_length for j in range(self.args.image_sequence_length)]
            image_primary = [image_primary[:, idx, ...].unsqueeze(1) for idx in indices]
            image_primary = torch.cat(image_primary, dim=1)

            image_wrist = [image_wrist[:, idx, ...].unsqueeze(1) for idx in indices]
            image_wrist = torch.cat(image_wrist, dim=1)

            state = [state[:, idx, ...].unsqueeze(1) for idx in indices]
            state = torch.cat(state, dim=1)
            state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        sub_primary = torch.cat(sub_primary, dim=1)
        sub_wrist = torch.cat(sub_wrist, dim=1)
        sub_state = torch.cat(sub_state, dim=1)
        sub_actions = torch.cat(sub_actions, dim=1)
        # print(sub_actions.shape, sub_primary.shape, sub_wrist.shape, sub_state.shape) torch.Size([1, 300, 7]) torch.Size([1, 300, 3, 224, 224]) torch.Size([1, 300, 3, 224, 224]) torch.Size([1, 300, 15])

        sample_interval = self.args.ttt_traj_len // self.args.ttt_sample_repeat
        for chunk_idx in range(self.args.ttt_sample_repeat):
            # random_choose = random.randint(self.args.image_sequence_length, self.args.ttt_traj_len-self.args.window_size-2)
            random_choose = min(self.args.image_sequence_length + chunk_idx * sample_interval, self.args.ttt_traj_len-self.args.window_size-2)
            
            # Extract the data for this chunk
            actions = sub_actions[:, random_choose:random_choose+self.args.window_size, ...]
            image_primary_chunk = sub_primary[:, random_choose:random_choose+self.args.window_size, ...]
            image_wrist_chunk = sub_wrist[:, random_choose:random_choose+self.args.window_size, ...]
            state_chunk = sub_state[:, random_choose:random_choose+self.args.window_size, ...]
            text_chunk = no_text.unsqueeze(1).repeat(1, self.num_chunk, 1)
            # print(actions.shape) torch.Size([1, 23, 7])
            # print(image_primary_chunk.shape) torch.Size([1, 23, 3, 224, 224])
            # print(state_chunk.shape) torch.Size([1, 23, 15])

            # Save each trajectory as a separate .npy file
            trajectory_data = {
                'actions': actions.cpu().numpy(),
                'image_primary': image_primary_chunk.cpu().numpy(),
                'image_wrist': image_wrist_chunk.cpu().numpy(),
                'state': state_chunk.cpu().numpy(),
                'text': text_chunk.cpu().numpy(), 
            }
            save_path = save_dir / f"test_{self.test_idx}_data_{traj_idx:06d}.npy"
            np.save(save_path, trajectory_data)
            traj_idx += 1
        
        return traj_idx, trajectory_data
    
    def sample(self):
        with open(self.eval_seq_path, 'r') as f:
            eval_sequences = json.load(f)
            # random.shuffle(eval_sequences)
            initial_state, tasks = eval_sequences[self.test_idx]
        with open(self.lang_path, 'r') as f:
            tasks = json.load(f)
        conf_dir = Path(self.args.calvin_conf_path)
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        save_dir = Path(self.args.ttt_data_dir)
        save_dir.mkdir(exist_ok=True)
        traj_idx = 0
        self.reset_env(initial_state)
        random.seed(self.args.seed)
        assert self.args.ttt_num_samples <= len(tasks)
        for i in tqdm(range(len(tasks)), desc="Sampling"):
            self.reset_env(initial_state)
            subtask = tasks[i]
            traj_idx, _ = self.sample_one_traj(val_annotations, subtask, save_dir, traj_idx)
        print(f"{traj_idx} Traj are Sampled in the Test Time Training Stage for Test {self.test_idx}.")
        return traj_idx
    
    def get_robot_info_from_multi_step_env(self, actions, image_process_fn):
        B, action_seq_len, _ = actions.shape
        primary_list = []
        wrist_list = []
        state_list = []
        current_info_list = []
        for i in range(action_seq_len):
            action = actions[0, i, ...].numpy()
            obs, _, _, curren_info = self.env.step(action)
            state_list.append(torch.from_numpy(obs["robot_obs"]).unsqueeze(0).unsqueeze(1))
            primary_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_static"])]).unsqueeze(0))
            wrist_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_gripper"])]).unsqueeze(0))
            current_info_list.append(curren_info)
        primary = torch.cat(primary_list, dim=1)
        wrist = torch.cat(wrist_list, dim=1)
        states = torch.cat(state_list, dim=1)
        return primary, wrist, states, current_info_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestTimeTrainingDataset(Dataset):
    def __init__(self, args, test_idx):
        data_dir = Path(args.ttt_data_dir)
        if args.ttt_use_all_data:
            lens = len(list(data_dir.glob(f"test_*_data_*.npy")))
            self.trajectory_files = random.sample(list(data_dir.glob(f"test_*_data_*.npy")), lens)
        else:
            self.trajectory_files = sorted(list(data_dir.glob(f"test_{test_idx}_data_*.npy")))[:args.ttt_num_samples*args.ttt_sample_repeat]

    def __len__(self):
        return len(self.trajectory_files)
    
    def __getitem__(self, idx):
        traj_data = np.load(self.trajectory_files[idx], allow_pickle=True).item()
        
        return {
            'image_primary': torch.from_numpy(traj_data['image_primary']),
            'image_wrist': torch.from_numpy(traj_data['image_wrist']),
            'states': torch.from_numpy(traj_data['state']),
            'input_texts': torch.from_numpy(traj_data['text']),
            'input_actions': torch.from_numpy(traj_data['actions']),
        }

def create_dataloader(args, test_idx, shuffle=True, num_workers=4):
    dataset = TestTimeTrainingDataset(args, test_idx)
    return DataLoader(
        dataset,
        batch_size=args.ttt_batch_size,
        shuffle=shuffle,
        collate_fn=trajectory_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

def trajectory_collate_fn(batch):
    result = {
        'image_primary': [],
        'image_wrist': [],
        'states': [],
        'input_texts': [],
        'input_actions': []
    }
    
    for item in batch:
        for key in result:
            result[key].append(item[key])
    
    result['image_primary'] = torch.cat(result['image_primary'], dim=0)
    result['image_wrist'] = torch.cat(result['image_wrist'], dim=0)
    result['states'] = torch.cat(result['states'], dim=0)
    result['input_texts'] = torch.cat(result['input_texts'], dim=0)
    result['input_actions'] = torch.cat(result['input_actions'], dim=0)

    return result


class TestTimeTrainer(nn.Module):
    def __init__(self, 
    env,
    args,
    image_process_fn,
    text_process_fn,
    wrapped_model,
    eval_seq_path,
    lang_path,
    test_idx,
    ):
        super().__init__()
        self.init_model = wrapped_model.model
        self.wrapped_model = wrapped_model
        self.args = args
        self.test_idx = test_idx
        set_seed(self.args.seed)
        self.device = 'cuda'
        self.text_process_fn = text_process_fn
        self.image_process_fn = image_process_fn
        # if os.path.exists(self.args.ttt_data_dir):    
        #     shutil.rmtree(self.args.ttt_data_dir)
        os.makedirs(self.args.ttt_data_dir, exist_ok=True)
        self.gradient_accumulation_steps = args.ttt_num_samples * args.ttt_sample_repeat
        if not self.args.use_sampled_data:
            randomsampler = RandomSampler(self.wrapped_model, env, args, eval_seq_path, lang_path, self.text_process_fn, self.image_process_fn, self.test_idx)
            randomsampler.sample()
        backbone = "qwen3" if self.args.use_qwen else "gpt2"
        self.lora = LoraModel(self.init_model, lora_rank=self.args.lora_rank, lora_alpha=self.args.lora_alpha, \
                              lora_dropout=self.args.lora_dropout, lora_mode=self.args.lora_mode, backbone=backbone)
        self.lora.to(device=self.device)
        self.lora_model = self.lora._apply_finetuning_to_gpt2()
        self.wrapped_model.update_model(self.lora_model)
        self.wrapped_model.model.to(self.device)

        self.dataloader = create_dataloader(args, self.test_idx)
        num_samples = len(self.dataloader)
        self.num_batches = math.floor(num_samples / self.args.batch_size)
        self.warmup_steps = self.num_batches * args.warmup_epochs
        self.cast_dtype = get_cast_dtype(args.precision)

    def trainer(self):
        self.init_model.to(self.device)
        self.init_model.train()

        trainable_params = self.lora._get_trainable_parameters()

        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.ttt_learning_rate, weight_decay=self.args.ttt_weight_decay)

        epoch_pbar = tqdm(range(self.args.ttt_num_epoch), desc="TTT Epoch", position=0)
        num_batches_per_epoch = self.num_batches
        total_training_steps = num_batches_per_epoch * self.args.ttt_num_epoch 
        lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_training_steps,
            )
        autocast = get_autocast(self.args.precision)
        num_chunk = self.args.sequence_length

        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"epoch {epoch+1}/{self.args.ttt_num_epoch}")
            
            batch_pbar = tqdm(
                    enumerate(self.dataloader),
                    desc=f"Batch (Epoch {epoch+1})")
            mv_avg_loss = []
            for num_steps, batch in batch_pbar:
                
                image_primary = batch['image_primary'].to(self.device, dtype=self.cast_dtype, non_blocking=True)
                image_wrist = batch['image_wrist'].to(self.device, dtype=self.cast_dtype, non_blocking=True)
                states = batch['states'].to(self.device, dtype=self.cast_dtype, non_blocking=True)
                states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
                states[..., 6:] = (states[..., 6:] + 1) // 2

                
                input_texts = batch['input_texts'].to(self.device, non_blocking=True)
                input_actions = batch['input_actions'].to(self.device, dtype=self.cast_dtype, non_blocking=True)
                input_actions[..., 6:] = (input_actions[..., 6:] + 1) // 2

                num_chunk = self.args.sequence_length


                input_image_primary, label_image_primary = self.sample(image_primary, self.args, num_chunk)
                input_image_wrist, label_image_wrist = self.sample(image_wrist, self.args, num_chunk)
                input_states, label_states = self.sample(states, self.args, num_chunk)

                input_texts = input_texts[:, :num_chunk, ...]

                input_actions = input_actions[:, self.args.image_sequence_length:, ...]
                label_actions = input_actions
                
                with autocast():  # image_primary, image_wrist, state, language_instruction
                    if self.args.eval_action_entropy:
                        _, _, image_pred, arm_pred_state, gripper_pred_state, _ = self.init_model(
                            image_primary=input_image_primary,
                            image_wrist=input_image_wrist,
                            state=input_states,
                            text_token=input_texts,
                            action=input_actions,
                        )
                    else:
                        _, _, image_pred, arm_pred_state, gripper_pred_state = self.init_model(
                            image_primary=input_image_primary,
                            image_wrist=input_image_wrist,
                            state=input_states,
                            text_token=input_texts,
                            action=input_actions,
                        )
                label_image_primary = patchify(label_image_primary.flatten(0, 1), patch_size=self.args.patch_size)
                label_image_wrist = patchify(label_image_wrist.flatten(0, 1), patch_size=self.args.patch_size)
                # label_image_primary = normalize_patchfied_image(label_image_primary)
                # label_image_wrist = normalize_patchfied_image(label_image_wrist)
                loss_image = 0.5 * (torch.nn.functional.mse_loss(   
                            image_pred[:, 0, :, :], 
                            label_image_primary.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 1, :, :], 
                            label_image_wrist.detach()))

                if self.args.loss_state:
                    loss_arm_state = torch.nn.functional.smooth_l1_loss(
                                arm_pred_state.flatten(1, 2), 
                                label_states[..., :6].detach())
                    
                    loss_gripper_state = torch.nn.functional.binary_cross_entropy(
                                    gripper_pred_state.flatten(1, 2), 
                                    label_states[..., 6:].detach())
                else:
                    loss_arm_state = torch.tensor([0.0]).to(self.device)
                    loss_gripper_state = torch.tensor([0.0]).to(self.device)
                
                # loss_ttt = 0.1 * loss_image + 0.01 * loss_arm_state + 0.01 * loss_gripper_state
                loss_ttt = 0.1 * loss_image
                loss = loss_ttt / self.gradient_accumulation_steps
                mv_avg_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lora_model.parameters(), self.args.ttt_max_grad_norm)

                if (((num_steps + 1) % self.gradient_accumulation_steps) == 0) or (
                    num_steps == num_batches_per_epoch - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
                avg_horizon = min(100, len(mv_avg_loss))
                batch_pbar.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss_image": loss_image.item(), "loss_arm_state": loss_arm_state.item(), "loss_gripper_state": loss_gripper_state.item()})
        self.wrapped_model.update_model(self.init_model)
        return self.wrapped_model
    
    def save_checkpoint(self, epoch):
        save_dir = "./checkpoints/ttt_uniaorld"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        
        state_dict = {
        'model': self.wrapped_model.model.state_dict(),
        }
        
        torch.save(state_dict, save_path)
        
        print(f"Checkpoint saved to {save_path}")

    def sample(self, input_tensor, args, num_chunk):
        input_list = []
        label_list = []
        input_list.append(input_tensor[:, :args.image_sequence_length, ...])
        for i in range(num_chunk):
            block = input_tensor[:, args.image_sequence_length+i*args.action_sequence_length:args.image_sequence_length+(i+1)*args.action_sequence_length, ...]
            # L = args.action_sequence_length
            # if args.image_sequence_length == 1:
            #     indices = [L - 1]
            # else:
            # indices = [(j * args.action_sequence_length) // args.image_sequence_length for j in range(args.image_sequence_length)]

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
    
    def add_noise_to_text(self, input_texts, noise_factor=0.1):
        float_tensor = input_texts.float()
        noise = torch.randn_like(float_tensor) * noise_factor
        noisy_float = float_tensor + noise
        noisy_int = torch.round(noisy_float).to(torch.int32)

        return noisy_int

def unpatchify(x, patch_size):

    """
    x: (N, L, patch_size**2 *3)  / (B, N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)  / (B, N, 3, H, W)
    """
    import math
    if len(x.shape) == 4:  # (B, N, L, patch_size**2 * 3)
        h = w = int(math.sqrt(x.shape[2]))
        assert h * w == x.shape[2]
        
        # 重塑为 (B, N, h, w, patch_size, patch_size, 3)
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, patch_size, patch_size, 3))
        # 调整维度顺序
        x = torch.einsum('bnhwpqc->bnchpwq', x)
        # 最后重塑为 (B, N, 3, H, W)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], 3, h * patch_size, w * patch_size))
    else:  # (N, L, patch_size**2 * 3)
        h = w = int(math.sqrt(x.shape[1]))
        assert h * w == x.shape[1]
        
        # 重塑为 (N, h, w, patch_size, patch_size, 3)
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        # 调整维度顺序
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 最后重塑为 (N, 3, H, W)
        imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
    
    return imgs

def save_video_as_gif(video_tensor, output_path="video.gif", fps=8):
    """
    Save a video tensor as a GIF.
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape [B, T, C, H, W] or [T, C, H, W]
                                     B: batch size, T: frames, C: channels, H: height, W: width
        output_path (str): Path to save the GIF
        fps (int): Frames per second for the GIF
    """
    # Remove batch dimension if present
    if len(video_tensor.shape) == 5:
        video_tensor = video_tensor.squeeze(0)  # Remove batch dimension, resulting in [T, C, H, W]
    
    # Ensure tensor is on CPU and convert to numpy
    video_np = video_tensor.cpu().detach().numpy()
    
    # Convert from [T, C, H, W] to [T, H, W, C] format
    video_np = np.transpose(video_np, (0, 2, 3, 1))
    
    # Normalize pixel values to [0, 255] if they're in [0, 1] range
    video_np = (video_np * 255).astype(np.uint8)
    
    # Create a list of PIL images
    pil_images = [Image.fromarray(frame) for frame in video_np]
    
    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        optimize=False,
        duration=int(1000/fps),  # milliseconds between frames
        loop=0  # 0 means loop indefinitely
    )
    
    print(f"GIF saved to {output_path}")
    
    return output_path
    