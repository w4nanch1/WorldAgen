from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import copy
import copy
from collections import deque
from models.model import UniAorld
# from moviepy.editor import ImageSequenceClip
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
from tqdm.auto import tqdm
import copy
from calvin_env.envs.play_table_env import get_env
from utils.calvin_data_utils import preprocess_image, preprocess_text_calvin
import functools
from omegaconf import OmegaConf
from utils.train_utils import get_cast_dtype
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from typing import Callable, Any
from utils.train_utils import get_checkpoint



os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000
INTERVAL = 1

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    return env

def add_noise_to_action(action):
    if isinstance(action, torch.Tensor):
        action = action.numpy()
    lens = action.shape[-1]
    noisy_action = action.copy()
    noise = np.random.uniform(-1, 1, lens-1)
    noisy_action[:lens-1] += noise
    noisy_action[lens-1] = np.random.choice([-1, 1])
    noisy_action = torch.from_numpy(noisy_action)
    return noisy_action

def add_noise_to_state(state):
    if isinstance(state, torch.Tensor):
        state = state.numpy()
    lens = state.shape[-1]
    noisy_state = state.copy()
    noise = np.random.uniform(-1, 1, lens-1)
    noisy_state[:lens-1] += noise
    noisy_state[lens-1] = np.random.choice([-1, 1])
    return noisy_state

def unpatchify(x, batch_size=1, img_size=224, patch_size=16):
    seq_batch, num_patches, patch_dim = x.shape
    patches_per_side = img_size // patch_size
    channels = patch_dim // (patch_size * patch_size)
    seq_len = seq_batch // batch_size
    
    x = x.reshape(batch_size, seq_len, num_patches, channels, patch_size, patch_size)
    x = x.reshape(batch_size, seq_len, patches_per_side, patches_per_side, channels, patch_size, patch_size)

    x = x.permute(0, 1, 4, 2, 5, 3, 6)
    x = x.reshape(batch_size, seq_len, channels, img_size, img_size)
    return x

def sample_split_as_chunk(input_tensor, args, num_chunk):
    input_list = []
    label_list = []
    input_list.append(input_tensor[:, :args.image_sequence_length, ...].unsqueeze(1))
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
        samples = torch.cat(samples, dim=1).unsqueeze(1)
        if i != num_chunk - 1:
            input_list.append(samples)
        label_list.append(samples)
    # print(samples.shape) torch.Size([1, 1, 3, 3, 224, 224])
    input_list = torch.cat(input_list, dim=1)
    label_list = torch.cat(label_list, dim=1)
    return input_list, label_list

class UniAorldModelWrapper(CalvinBaseModel):
    def __init__(self, args, model: UniAorld, tokenizer, image_processor, cast_dtype, history_len=10,
                calvin_eval_max_steps=360):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.use_diff = False
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.calvin_eval_max_steps = calvin_eval_max_steps
        self.device = "cuda"
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.action_queue = deque(maxlen=history_len)
        self.args = args

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.action_queue = deque(maxlen=self.history_len)

    def update_model(self, new_model):
        self.model = new_model

    def reset_text(self):
        text0 = self.text_queue[0]
        print("Now text queue item is:", text0)
        self.text_queue = deque(maxlen=self.history_len)

        for _ in range(self.history_len):
            self.text_queue.append(torch.zeros_like(text0).to(self.device))

    def step_agent(self, text, state, image_primary, image_wrist, action=None):
        text = text.unsqueeze(1)
            
        if action != None:
            action = action.unsqueeze(1).to(dtype=self.cast_type)
            action = torch.cat([action[..., :6], action[..., [-1]]], dim=-1)
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        image_primary = image_primary.unsqueeze(1).to(dtype=self.cast_type) # [B, 1, image_seq_len, ...]
        image_wrist = image_wrist.unsqueeze(1).to(dtype=self.cast_type)

        # print("image_primary", image_primary.shape) torch.Size([1, 1, 3, 3, 224, 224])
        # print("state", state.shape) # torch.Size([1, 1, 3, 7])
        with torch.no_grad():
            text = text.to(self.device)
            state = state.to(self.device)
            if action != None:
                action = action.to(self.device)
            image_primary = image_primary.to(self.device)
            image_wrist = image_wrist.to(self.device)

            self.img_queue.append(image_primary)
            self.gripper_queue.append(image_wrist)
            self.state_queue.append(state)

            if len(self.text_queue) == 0 and text is not None:
                for _ in range(self.history_len):
                    self.text_queue.append(text)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            # print("img", image_primary.shape) torch.Size([1, 1, 3, 3, 224, 224])
            # print("state", state.shape) torch.Size([1, 1, 3, 7])
            action_deque_empty = False
            if len(self.action_queue) == 0:
                action_deque_empty = True
                self.action_queue.append(action)
            action = torch.cat(list(self.action_queue), dim=1)
            
            num_step = image_primary.shape[1]
            if num_step < self.history_len:  
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1, 1, 1)], dim=1)
                # print(input_image_primary.shape) torch.Size([1, 4, 3, 3, 224, 224])
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            num_action = action.shape[1]
            if num_action < self.history_len:
                input_action = torch.cat([action, action[:, -1].unsqueeze(1).repeat(1, self.history_len-num_action, 1, 1)], dim=1)
            else:
                input_action = action

            B = input_action.shape[0]
            num_chunk = self.args.sequence_length
            action_seq_len = input_action.shape[2]
            arm_pred_action, gripper_pred_action, _, _, _ = self.model(
                image_primary=input_image_primary.flatten(1, 2),
                image_wrist=input_image_wrist.flatten(1, 2),
                state=input_state.flatten(1, 2),
                text_token=input_text_token,
                action=input_action.flatten(1, 2)
            )
            # print("+++++++++++++++++++++++++++")
            if action_deque_empty:
                self.action_queue = deque(maxlen=self.history_len)
                action_deque_empty = False
            # print(arm_pred_action.shape, gripper_pred_action.shape) # torch.Size([1, 2, 5, 6]) torch.Size([1, 2, 5, 1])
            action = torch.concat((arm_pred_action[0, :, :self.args.action_sequence_length, :], gripper_pred_action[0, :, :self.args.action_sequence_length, :] > 0.5), dim=-1)
            action[..., -1] = (action[..., -1] - 0.5) * 2  # scale to -1 or 1
            action = action.view(B, num_chunk, action_seq_len, 7)
            if num_step < self.history_len:
                action = action[:, num_step - 1, ...]
            else:
                action = action[:, -1, ...]
            action = action.cpu().detach().to(dtype=torch.float16)
        return action
            
    def step_world_model(self, action):
        action = action.unsqueeze(1).to(dtype=self.cast_type)
        action = torch.cat([action[..., :6], action[..., [-1]]], dim=-1)
        with torch.no_grad():
            action = action.to(self.device)

            self.action_queue.append(action)

            input_image_primary = torch.cat(list(self.img_queue), dim=1)
            input_image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            input_state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            input_action = torch.cat(list(self.action_queue), dim=1)

            num_step = input_image_primary.shape[1]
            if num_step < self.history_len:  
                input_image_primary = torch.cat([input_image_primary, input_image_primary[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([input_image_wrist, input_image_wrist[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1, 1, 1)], dim=1)
                input_state = torch.cat([input_state, input_state[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1)], dim=1)
            num_action = input_action.shape[1]
            if num_action < self.history_len:
                input_action = torch.cat([input_action, input_action[:, -1].unsqueeze(1).repeat(1, self.history_len-num_action, 1, 1)], dim=1)

            B = input_image_primary.shape[0]
            num_chunk = self.args.sequence_length
            image_seq_len = input_image_primary.shape[2]
            state_seq_len = input_state.shape[2]

            _, _, image_pred, arm_pred_state, gripper_pred_state = self.model(
                image_primary=input_image_primary.flatten(1, 2),
                image_wrist=input_image_wrist.flatten(1, 2),
                state=input_state.flatten(1, 2),
                text_token=input_text_token,
                action=input_action.flatten(1, 2),
            )
            # print(image_pred.shape, arm_pred_state.shape, gripper_pred_state.shape) torch.Size([6, 2, 196, 768]) torch.Size([1, 2, 3, 6]) torch.Size([1, 2, 3, 1])
            # pred_image_primary 6, 196, 768
            state = None
            if arm_pred_state is not None and gripper_pred_state is not None:
                state = torch.concat((arm_pred_state[0, :, :, :], gripper_pred_state[0, :, :, :] > 0.5), dim=-1)
                state = state.view(B, num_chunk, state_seq_len, 7)
                if num_step < self.history_len:
                    state = state[:, num_step - 1, ...]
                else:
                    state = state[:, -1, ...]
                state = state.cpu().detach().to(dtype=torch.float16)
            pred_image_primary, pred_image_wrist = None, None
            if image_pred is not None:
                pred_image_primary = image_pred[:, 0, :, :]
                pred_image_wrist = image_pred[:, 1, :, :]
                pred_image_primary = unpatchify(pred_image_primary).unflatten(1, (num_chunk, image_seq_len))
                pred_image_wrist = unpatchify(pred_image_wrist).unflatten(1, (num_chunk, image_seq_len))

                if num_step < self.history_len:
                    pred_image_primary = pred_image_primary[:, num_step - 1, ...].cpu().detach().to(dtype=torch.float32)
                    pred_image_wrist = pred_image_wrist[:, num_step - 1, ...].cpu().detach().to(dtype=torch.float32)
                else:
                    pred_image_primary = pred_image_primary[:, -1, ...].cpu().detach().to(dtype=torch.float32)
                    pred_image_wrist = pred_image_wrist[:, -1, ...].cpu().detach().to(dtype=torch.float32)
            return pred_image_primary, pred_image_wrist, state


def evaluate_policy_ddp(model, env, args, epoch, calvin_conf_path, text_process_fn, image_process_fn,  eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    cast_dtype = get_cast_dtype(args.precision)
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('./utils/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('./utils/eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    import copy
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    results_ttt = []
    plans = defaultdict(list)
    
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    
    n = 0

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if args.test_ori:
        local_sequence_i = 0
        env = make_env(args.calvin_dataset)
        for initial_state, eval_sequence in eval_sequences:
            # import pybullet as p
            # print("after make_env pybullet connections:", p.getConnectionInfo())
            try:
                result, all_pred_image_primary_list, all_pred_image_wrist_list, all_pred_state_list, all_img_pri_sim, all_img_wrist_sim \
                    = evaluate_sequence(env, args, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, text_process_fn, image_process_fn, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
            finally:    
                # print("before close pybullet connections:", p.getConnectionInfo())
                # env.close()
                # del env
                pass
                # print("after close pybullet connections:", p.getConnectionInfo())
            # tensors_to_gif(all_pred_image_primary_list, f'./pred_pri_sim_{n}.gif')
            # tensors_to_gif(all_img_pri_sim, f'./pri_sim_{n}.gif')
            # tensors_to_gif(all_pred_image_wrist_list, f'./pred_wrist_sim_{n}.gif')
            # tensors_to_gif(all_img_wrist_sim, f'./wrist_sim_{n}.gif')
            n+=1
            results.append(result)
            eval_sequences.set_description(" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]))
            local_sequence_i += 1
            
        if create_plan_tsne:
            create_tsne(plans, eval_log_dir, epoch)

        eval_sequences = extract_iter_from_tqdm(eval_sequences)

        for re in [results]:
            res_tup = [(res, eval_seq) for res, eval_seq in zip(re, eval_sequences)]
            all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
            torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

            if torch.distributed.get_rank() == 0:
                res_tup_list = merge_multi_list(all_res_tup)
                res_list = [_[0] for _ in res_tup_list]
                eval_seq_list = [_[1] for _ in res_tup_list]
                print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)
        
    if args.ttt: 
        p=0
        local_sequence_i = 0
        ttt_model = None
        for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
            if idx == 0 or (idx > 0 and args.ttt_train_every_test_sample):
                if ttt_model is not None:
                    del ttt_model
                    del ttt_wrapped_model  
                    del ttt_trainer
                    torch.cuda.empty_cache()
                env = make_env(args.calvin_dataset)
                ttt_model = UniAorld(
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
                )
                ttt_model = ttt_model.to(device_id)
                if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
                    ttt_model = ttt_model.bfloat16()
                elif args.precision == "fp16":
                    ttt_model = ttt_model.half()
                elif args.precision == "fp32":
                    ttt_model = ttt_model.float()
                    if 'vision_encoder' in args.bf16_module:
                        ttt_model.vision_encoder.bfloat16()
                    if "causal_transformer" in args.bf16_module:
                        ttt_model.transformer_backbone.bfloat16()
                    if "image_decoder" in args.bf16_module:
                        ttt_model.image_decoder.bfloat16()
                        ttt_model.image_decoder_obs_pred_projector.bfloat16()
                ttt_model._init_model_type()
                ttt_model = DDP(ttt_model, device_ids=[device_id], find_unused_parameters=True)
                ttt_model.load_state_dict(checkpoint["model_state_dict"], False)

                ttt_wrapped_model = UniAorldModelWrapper(
                                args,
                                ttt_model, 
                                text_process_fn, 
                                image_process_fn, 
                                cast_dtype, 
                                history_len=args.sequence_length, 
                                calvin_eval_max_steps=EP_LEN)

                from test_time_training.ttt import TestTimeTrainer
                ttt_trainer = TestTimeTrainer(env, 
                                        args,
                                        image_process_fn,
                                        text_process_fn,
                                        ttt_wrapped_model, 
                                        eval_seq_path='./utils/eval_sequences.json',
                                        lang_path='./utils/all_lang.json',
                                        test_idx=local_sequence_i
                                    )
                # ttt_model = ttt_trainer.trainer()
                if not args.ttt_load_model:
                    ttt_model = ttt_trainer.trainer()
                    if args.save_ttt_ckpt:
                        ckpt_dir = "./checkpoints/lora/"
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model_name = str(args.lora_rank) + 'rank_' + str(args.lora_alpha) + 'alpha_' + str(args.ttt_num_samples) + "samples_" + str(args.ttt_sample_repeat) + "repeat_" + str(args.ttt_learning_rate) + "lr_" + str(args.ttt_num_epoch) + "epoch.pth"
                        ckpt_path = os.path.join(ckpt_dir, model_name)
                        print(f"Saving lora checkpoint to {ckpt_path}")
                        checkpoint_data = {
                            'model_state_dict': ttt_model.model.state_dict(),
                            'model_config': {
                                'lora_rank': args.lora_rank,
                                'lora_alpha': args.lora_alpha,
                                'lora_dropout': args.lora_dropout,
                                'lora_mode': args.lora_mode,
                            }
                        }
                        torch.save(checkpoint_data, ckpt_path)
                    else:
                        model_name = str(args.lora_rank) + 'rank_' + str(args.lora_alpha) + 'alpha_' + str(args.ttt_num_samples) + "samples_" + str(args.ttt_sample_repeat) + "repeat_" + str(args.ttt_learning_rate) + "lr_" + str(args.ttt_num_epoch) + "epoch.pth"
                        print(f"Finishing lora training {model_name}, but not save")
                else:
                    ttt_model = ttt_trainer.wrapped_model
                    try:
                        checkpoint_data = torch.load(args.ttt_load_model_path, map_location=f'cuda:{device_id}')
                        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                            state_dict = checkpoint_data['model_state_dict']
                            print(f"Loading checkpoint with config: {checkpoint_data.get('model_config', 'No config info')}")
                        else:
                            state_dict = checkpoint_data
                            print("Loading checkpoint in old format (state_dict only)")
                        
                        missing_keys, unexpected_keys = ttt_model.model.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                            print(f"Warning: Missing keys when loading checkpoint: {missing_keys}")
                        if unexpected_keys:
                            print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")
                        print(f"Successfully loaded checkpoint from {args.ttt_load_model_path}")
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
                        raise
                ttt_model.model.eval()
            result_ttt, all_pred_image_primary_list_ttt, all_pred_image_wrist_list_ttt, all_pred_state_list_ttt, all_img_pri_sim_ttt, all_img_wrist_sim_ttt \
                = evaluate_sequence(env, args, ttt_model, task_oracle, initial_state, eval_sequence, val_annotations,  plans, text_process_fn, image_process_fn, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
            # tensors_to_gif(all_pred_image_primary_list_ttt, f'./pred_pri_sim_{p}_ttt.gif')
            # tensors_to_gif(all_img_pri_sim_ttt, f'./pri_sim_{p}_ttt.gif')
            # tensors_to_gif(all_pred_image_wrist_list_ttt, f'./pred_wrist_sim_{p}_ttt.gif')
            # tensors_to_gif(all_img_wrist_sim_ttt, f'./wrist_sim_{p}_ttt.gif')
            p += 1
            results_ttt.append(result_ttt)
            eval_sequences.set_description(" TTT: " + " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results_ttt))]))
            # log_metrics_to_file(results, results_ttt, log_file_path=f"evaluation_log_{args.lora_mode}.txt")
            local_sequence_i += 1
            if args.ttt_train_every_test_sample:
                env.close()
                del env
        
        if create_plan_tsne:
            create_tsne(plans, eval_log_dir, epoch)

        eval_sequences = extract_iter_from_tqdm(eval_sequences)

        for re in [results_ttt]:
            res_tup = [(res, eval_seq) for res, eval_seq in zip(re, eval_sequences)]
            all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
            torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

            if torch.distributed.get_rank() == 0:
                res_tup_list = merge_multi_list(all_res_tup)
                res_list = [_[0] for _ in res_tup_list]
                eval_seq_list = [_[1] for _ in res_tup_list]
                print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)
        model_name = str(args.lora_rank) + 'rank_' + str(args.lora_alpha) + 'alpha_' + str(args.ttt_num_samples) + "samples_" + str(args.ttt_sample_repeat) + "repeat_" + str(args.ttt_learning_rate) + "lr_" + str(args.ttt_num_epoch) + "epoch_" + str(args.ttt_traj_len) + "traj_len.pth"
        print(f"Finishing lora training {model_name}, but not save")

    return results, results_ttt

def evaluate_sequence(env, args, model, task_checker, initial_state, eval_sequence, val_annotations, plans, text_process_fn, image_process_fn, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    # print(robot_obs.shape, scene_obs.shape) (15,) (24,)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    all_pred_image_primary_list, all_pred_image_wrist_list, all_pred_state_list = [], [], []
    all_img_pri_smi, all_img_wrist_sim = [], []
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success, pred_image_primary_list, pred_image_wrist_list, pred_state_list, img_pri_sim, img_wrist_sim = rollout(env, args, model, task_checker, subtask, val_annotations, plans, text_process_fn, image_process_fn, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, robot_obs=robot_obs, scene_obs=scene_obs)
            all_pred_image_primary_list.extend(pred_image_primary_list)
            all_pred_image_wrist_list.extend(pred_image_wrist_list)
            all_pred_state_list.extend(pred_state_list)
            all_img_pri_smi.extend(img_pri_sim)
            all_img_wrist_sim.extend(img_wrist_sim)
        else:
            success, pred_image_primary_list, pred_image_wrist_list, pred_state_list, img_pri_sim, img_wrist_sim = rollout(env, args, model, task_checker, subtask, val_annotations, plans, text_process_fn, image_process_fn, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst)
            all_pred_image_primary_list.extend(pred_image_primary_list)
            all_pred_image_wrist_list.extend(pred_image_wrist_list)
            all_pred_state_list.extend(pred_state_list)
            all_img_pri_smi.extend(img_pri_sim)
            all_img_wrist_sim.extend(img_wrist_sim)
        if success:
            success_counter += 1
        else:
            return success_counter, all_pred_image_primary_list, all_pred_image_wrist_list, all_pred_state_list, all_img_pri_smi, all_img_wrist_sim
    return success_counter, all_pred_image_primary_list, all_pred_image_wrist_list, all_pred_state_list, all_img_pri_smi, all_img_wrist_sim

def rollout(env, args, model, task_oracle, subtask, val_annotations, plans, 
            text_process_fn, image_process_fn, 
            debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False, ):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """

    planned_actions = []
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()

    text = text_process_fn([lang_annotation])
    flag = False
    pred_image_primary_list, pred_image_wrist_list, pred_state_list = [], [], []
    img_pri_sim, img_wrist_sim = [], []
    for step in range(EP_LEN):
        if step == 0:
            image_primary = obs["rgb_obs"]['rgb_static']
            image_primary = Image.fromarray(image_primary)
            image_primary = image_process_fn([image_primary])
            image_primary = torch.cat([image_primary.unsqueeze(1) for _ in range(args.image_sequence_length)], dim=1)

            image_wrist = obs["rgb_obs"]['rgb_gripper']
            image_wrist = Image.fromarray(image_wrist)
            image_wrist = image_process_fn([image_wrist])
            image_wrist = torch.cat([image_wrist.unsqueeze(1) for _ in range(args.image_sequence_length)], dim=1)

            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state]))
            state = torch.cat([state.unsqueeze(1) for _ in range(args.state_sequence_length)], dim=1)

            action = torch.zeros(1, args.action_sequence_length, 7)
            action = model.step_agent(text, state, image_primary, image_wrist, action)
        else:
            action = model.step_agent(text, state, image_primary, image_wrist)
            
        pred_image_primary, pred_image_wrist, pred_state = model.step_world_model(action)
        pred_image_primary_list.append(pred_image_primary)
        pred_image_wrist_list.append(pred_image_wrist)
        pred_state_list.append(pred_state)

        image_primary, image_wrist, state, current_info_list = get_robot_info_from_multi_step_env(env, action, image_process_fn)
        # print(env.get_state_obs()['robot_obs'])
        # indices = [(j * args.action_sequence_length) // args.image_sequence_length for j in range(args.image_sequence_length)]
        if args.image_sequence_length == 1:
            indices = [args.action_sequence_length - 1]
        else:
        # indices = [(j * args.action_sequence_length) // args.image_sequence_length for j in range(args.image_sequence_length)]
            indices = torch.linspace(0, args.action_sequence_length-1, steps=args.image_sequence_length) \
                         .round() \
                         .long() \
                         .tolist()
        image_primary = [image_primary[:, idx, ...].unsqueeze(1) for idx in indices]
        image_primary = torch.cat(image_primary, dim=1)

        image_wrist = [image_wrist[:, idx, ...].unsqueeze(1) for idx in indices]
        image_wrist = torch.cat(image_wrist, dim=1)

        state = [state[:, idx, ...].unsqueeze(1) for idx in indices]
        state = torch.cat(state, dim=1)

        img_pri_sim.append(image_primary)
        img_wrist_sim.append(image_wrist)
        # check if current step solves a task
        for current_info in current_info_list:
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                flag = True
                return flag, pred_image_primary_list, pred_image_wrist_list, pred_state_list, img_pri_sim, img_wrist_sim
    return flag, pred_image_primary_list, pred_image_wrist_list, pred_state_list, img_pri_sim, img_wrist_sim

def test_calvin(args, model, dataloader, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False, ttt=False):
    # env = make_env(dataset_path)
    env = None
    device = 'cuda'
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
    wrapped_model = UniAorldModelWrapper(
                        args,
                        model, 
                        text_process_fn, 
                        image_process_fn, 
                        cast_dtype, 
                        history_len=hist_len, 
                        calvin_eval_max_steps=EP_LEN)
    
    evaluate_policy_ddp(wrapped_model, env, args, 0, args.calvin_conf_path, text_process_fn, image_process_fn,  eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)

def get_robot_info_from_multi_step_env(env, actions, image_process_fn):
    B, action_seq_len, _ = actions.shape
    primary_list = []
    wrist_list = []
    state_list = []
    current_info_list = []
    for i in range(action_seq_len):
        action = actions[0, i, ...].numpy()
        obs, _, _, curren_info = env.step(action)
        state_list.append(torch.from_numpy(obs["robot_obs"]).unsqueeze(0).unsqueeze(1))
        primary_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_static"])]).unsqueeze(0))
        wrist_list.append(image_process_fn([Image.fromarray(obs["rgb_obs"]["rgb_gripper"])]).unsqueeze(0))
        current_info_list.append(curren_info)
    primary = torch.cat(primary_list, dim=1)
    wrist = torch.cat(wrist_list, dim=1)
    states = torch.cat(state_list, dim=1)
    return primary, wrist, states, current_info_list

def tensors_to_gif(tensor_list, output_path="output.gif", fps=10):
    """
    Convert a list of PyTorch tensors to a GIF file.

    Args:
        tensor_list: List of PyTorch tensors of shape [1, 3, 3, 224, 224]
                     where dimensions are [batch, time, channels, height, width]
        output_path: Path to save the GIF
        fps: Frames per second for the GIF

    Returns:
        Path to the saved GIF
    """
    # Create a list to store the frames
    frames = []

    # Process each tensor in the list
    for tensor_idx, tensor in enumerate(tensor_list):
        for i in range(tensor.shape[1]):
            frame = tensor[0, i]

            frame_min_debug = frame.min().item()
            frame_max_debug = frame.max().item()
            # print(f"Tensor {tensor_idx}, Frame {i}: Original Min value = {frame_min_debug:.4f}, Max value = {frame_max_debug:.4f}")

            # Convert to numpy and transpose from [C, H, W] to [H, W, C]
            frame_np = frame.detach().cpu().numpy().transpose(1, 2, 0)

            current_min = np.min(frame_np)
            current_max = np.max(frame_np)

            # Linearly scale the data into the [0, 1] range.
            # Avoid division by zero; if max equals min, set all pixels to 0.
            if current_max == current_min:
                normalized_frame_np = np.zeros_like(frame_np)
            else:
                normalized_frame_np = (frame_np - current_min) / (current_max - current_min)
            
            # Scale the [0, 1] data to [0, 255] and convert to uint8.
            final_frame_np = (normalized_frame_np * 255).astype(np.uint8)
            
            # Convert to PIL Image
            frame_pil = Image.fromarray(final_frame_np)
            
            # Append to frames list
            frames.append(frame_pil)
    
    # Create the GIF
    imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"GIF saved to {output_path}")
    return output_path