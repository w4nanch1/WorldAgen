import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

from pathlib import Path
import copy
import io
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_utils import preprocess_text_calvin
import distutils.dir_util
import numpy as np
import time
import torch
from torch.distributed import gather
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype
import json
# libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from pdb import set_trace

def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    
    return euler

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

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

class UniAorldLiberoModelWrapper:
    def __init__(self, args, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                libero_eval_max_steps=600):
        super().__init__()
        self.model = model
        # Ensure model is on the correct device from the start
        self.device = "cuda"
        self.model = self.model.to(self.device)
        self.cast_type = cast_dtype
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.action_queue = deque(maxlen=history_len)
        self.args = args
        self.gripper_state = np.array([-1.0])
        self.cnt = 0

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.action_queue = deque(maxlen=self.history_len)
        self.gripper_state = np.array([-1.0])
        self.cnt += 1

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

        image_primary = image_primary.unsqueeze(1).to(dtype=self.cast_type)
        image_wrist = image_wrist.unsqueeze(1).to(dtype=self.cast_type)

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

            action_deque_empty = False
            if len(self.action_queue) == 0:
                action_deque_empty = True
                self.action_queue.append(action)
            action = torch.cat(list(self.action_queue), dim=1)
            
            num_step = image_primary.shape[1]
            if num_step < self.history_len:  
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].unsqueeze(1).repeat(1, self.history_len-num_step, 1, 1, 1, 1)], dim=1)
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


            if action_deque_empty:
                self.action_queue = deque(maxlen=self.history_len)
                action_deque_empty = False

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
                    action=input_action.flatten(1, 2)
                )
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

def get_robot_info_from_multi_step_env_libero(env, actions, image_process_fn):
    """Execute action chunk in LIBERO environment and collect observations"""
    B, action_seq_len, _ = actions.shape
    primary_list = []
    wrist_list = []
    state_list = []
    info_list = []
    success = 0
    
    for i in range(action_seq_len):
        action = actions[0, i, ...].numpy()
        try:
            obs, reward, done, info = env.step(action)
        except ValueError as e:
            if "terminated episode" in str(e):
                # Environment already terminated, break early
                break
            else:
                raise e
        
        # Process robot state (position + orientation + gripper)
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])
        gripper_state = np.array([action[-1]])  # Use action gripper state
        robot_state = np.concatenate([state_pos, state_ori, gripper_state])
        state_list.append(torch.from_numpy(robot_state).unsqueeze(0).unsqueeze(1))
        
        # Process images
        primary_image = Image.fromarray(obs["agentview_image"])
        wrist_image = Image.fromarray(obs["robot0_eye_in_hand_image"])
        primary_list.append(image_process_fn([primary_image]).unsqueeze(0))
        wrist_list.append(image_process_fn([wrist_image]).unsqueeze(0))
        
        info_list.append(info)
        
        if done:
            success = 1
            break
    
    # Handle case where no actions were executed due to early termination
    if len(primary_list) == 0:
        # Return dummy data with the same structure
        dummy_primary = torch.zeros(1, 1, 3, 224, 224)  # Assuming standard image size
        dummy_wrist = torch.zeros(1, 1, 3, 224, 224)
        dummy_state = torch.zeros(1, 1, 7)
        return dummy_primary, dummy_wrist, dummy_state, info_list, success
    
    primary = torch.cat(primary_list, dim=1)
    wrist = torch.cat(wrist_list, dim=1)
    states = torch.cat(state_list, dim=1)
    return primary, wrist, states, info_list, success

def evaluate_libero_task(task, env, obs, args, model):
    """Evaluate one LIBERO task using the UniAorld model with action/image chunk prediction"""
    steps = 0
    success = 0
    model.reset()
    goal = task.language
    
    # Preprocess initial text instruction
    text = model.text_process_fn([goal])
    with torch.no_grad():
        while steps < args.libero_eval_max_steps:
            if steps == 0:
                # Initial setup - prepare image, state, and action tensors
                image_primary = obs["agentview_image"]
                image_primary = Image.fromarray(image_primary)
                image_primary = model.image_process_fn([image_primary])
                image_primary = torch.cat([image_primary.unsqueeze(1) for _ in range(args.image_sequence_length)], dim=1)

                image_wrist = obs["robot0_eye_in_hand_image"]
                image_wrist = Image.fromarray(image_wrist)
                image_wrist = model.image_process_fn([image_wrist])
                image_wrist = torch.cat([image_wrist.unsqueeze(1) for _ in range(args.image_sequence_length)], dim=1)

                # Process robot state
                state_pos = obs["robot0_eef_pos"]
                state_ori = quaternion_to_euler(obs["robot0_eef_quat"])
                gripper_state = model.gripper_state
                robot_state = np.concatenate([state_pos, state_ori, gripper_state])
                state = torch.from_numpy(np.stack([robot_state]))
                state = torch.cat([state.unsqueeze(1) for _ in range(args.state_sequence_length)], dim=1)

                action = torch.zeros(1, args.action_sequence_length, 7)
                action = model.step_agent(text, state, image_primary, image_wrist, action)
            else:
                action = model.step_agent(text, state, image_primary, image_wrist)
            # Use world model to predict next state
            pred_image_primary, pred_image_wrist, pred_state = model.step_world_model(action)
            
            # Execute action chunk in environment and collect new observations
            image_primary, image_wrist, state, info_list, success = get_robot_info_from_multi_step_env_libero(
                env, action, model.image_process_fn)
            
            # Check if task completed successfully during action execution
            if success:
                env.close()
                return success
            
            # Sample from action sequence according to image sequence length
            if args.image_sequence_length == 1:
                indices = [args.action_sequence_length - 1]
            else:
                indices = torch.linspace(0, args.action_sequence_length-1, steps=args.image_sequence_length) \
                             .round() \
                             .long() \
                             .tolist()
            
            # Only sample if we have enough data
            actual_seq_len = image_primary.shape[1]
            if actual_seq_len > 0:
                # Adjust indices to not exceed actual sequence length
                valid_indices = [min(idx, actual_seq_len-1) for idx in indices]
                
                image_primary = [image_primary[:, idx, ...].unsqueeze(1) for idx in valid_indices]
                image_primary = torch.cat(image_primary, dim=1)

                image_wrist = [image_wrist[:, idx, ...].unsqueeze(1) for idx in valid_indices]
                image_wrist = torch.cat(image_wrist, dim=1)

                state = [state[:, idx, ...].unsqueeze(1) for idx in valid_indices]
                state = torch.cat(state, dim=1)
                
                # Update gripper state for next iteration
                model.gripper_state = np.array([action[0, min(valid_indices[-1], action.shape[1]-1), -1].item()])
            else:
                # If no observations were collected, break
                break
            
            steps += action.shape[1]  # Increment by action chunk length
            
    env.close()

def evaluate_policy_ddp(args, model, tokenizer, image_processor, cast_dtype):
    benchmark_dict = benchmark.get_benchmark_dict() 
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    results = []
    
    if "libero" in args.finetune_type:
        if args.finetune_type == "libero_10":
            global num_eval_episodes 
            global task_num
            num_eval_episodes = 20
            task_num = 10
             
            NUM_SEQUENCES = num_eval_episodes * task_num 
            eval_sequences = list(range(NUM_SEQUENCES))
            assert NUM_SEQUENCES % device_num == 0
            interval_len = int(NUM_SEQUENCES // device_num)
            eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
            eval_sequences = tqdm(eval_sequences)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.test_ori:
        for eval_id in eval_sequences:
            task_id = eval_id // num_eval_episodes
            exp_id = eval_id % num_eval_episodes 
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", task.problem_folder, task.bddl_file)
            env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.libero_img_size,
            "camera_widths": args.libero_img_size,
            "render_gpu_device_id":device_id
            }
            # print("device_id :", device_id)
            env = OffScreenRenderEnv(**env_args)
            env.task_id = task_id
            env.task_name = task_name
            env.task_suite_name = args.finetune_type
            env.reset()
            env.seed(args.seed)

            # set initial state
            init_states_path = os.path.join(
                f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            init_state = init_states[exp_id]
            obs = env.set_init_state(init_state)

            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros(7))
            result = evaluate_libero_task(task, env, obs, args, model)
            results.append(result) 
            print("rank", torch.distributed.get_rank(), "results :", results)
        
        def merge_multi_list(res):
            tmp = []
            for l in res:
                tmp.extend(l)
            return tmp

        def extract_iter_from_tqdm(tqdm_iter):
            return [_ for _ in tqdm_iter]

        eval_sequences = extract_iter_from_tqdm(eval_sequences)
        res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
        all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
        torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

        if torch.distributed.get_rank() == 0:
            res_tup_list = merge_multi_list(all_res_tup)
            res_tup_list.sort(key=lambda x: x[1])
            print_and_save(res_tup_list, task_suite)
    
    if args.ttt:
        from test_time_training.ttt_libero import (
            LiberoTestTimeTrainer)
        
        hist_len = args.sequence_length
        results_ttt = []
        
        # Group eval_sequences by task_id to process each scene only once
        task_groups = {}
        for eval_id in eval_sequences:
            task_id = eval_id // num_eval_episodes
            exp_id = eval_id % num_eval_episodes
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append((eval_id, exp_id))
        
        # Process tasks one by one to avoid memory accumulation
        for task_id, exp_list in task_groups.items():
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", 
                                        task.problem_folder, task.bddl_file)
            
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": args.libero_img_size,
                "camera_widths": args.libero_img_size,
                "render_gpu_device_id": device_id
            }
            
            print(f"=== Starting TTT for Task {task_id} ({task_name}) ===")
        
            # Use the first experiment for TTT training
            first_exp_id = exp_list[0][1]
            
            # Create environment for TTT training
            env = OffScreenRenderEnv(**env_args)
            env.task_id = task_id
            env.task_name = task_name
            env.task_suite_name = args.finetune_type
            env.reset()
            env.seed(args.seed)

            init_states_path = os.path.join(
                f"{args.libero_path}/libero/libero/init_files", 
                task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            init_state = init_states[first_exp_id]
            obs = env.set_init_state(init_state)

            for _ in range(5):
                env.step(np.zeros(7))

            # Clean up any existing models to free memory
            if 'ttt_model' in locals():
                del ttt_model
                torch.cuda.empty_cache()

            from models.model import UniAorld 
            
            # Create a fresh model for TTT
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
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
            ttt_model.load_state_dict(checkpoint["model_state_dict"], False)
            
            ttt_wrapped_model = UniAorldLiberoModelWrapper(
                args,
                ttt_model, 
                tokenizer, 
                image_processor, 
                cast_dtype, 
                history_len=hist_len, 
                libero_eval_max_steps=args.libero_eval_max_steps
            )

            # Perform TTT training using the first experiment
            ttt_trainer = LiberoTestTimeTrainer(
                args,
                functools.partial(preprocess_image, image_processor=image_processor),
                functools.partial(preprocess_text_calvin, tokenizer=tokenizer),
                ttt_wrapped_model, 
                task_id,
                first_exp_id  # Use first experiment for training
            )
            
            print(f"Training TTT model for Task {task_id} using Exp {first_exp_id}...")
            trained_ttt_model = ttt_trainer.trainer()
            trained_ttt_model.model.eval()
            
            print(f"TTT training completed for Task {task_id}. Evaluating all experiments...")
            
            # Close the training environment
            env.close()
            del env
            torch.cuda.empty_cache()
            # Now evaluate all experiments for this task using the trained model
            for eval_id, exp_id in exp_list:
                # Create fresh environment for evaluation
                eval_env = OffScreenRenderEnv(**env_args)
                eval_env.task_id = task_id
                eval_env.task_name = task_name
                eval_env.task_suite_name = args.finetune_type
                eval_env.reset()
                eval_env.seed(args.seed)

                init_state = init_states[exp_id]
                eval_obs = eval_env.set_init_state(init_state)

                for _ in range(5):
                    eval_env.step(np.zeros(7))

                # Use the trained model for this task
                evaluation_model = trained_ttt_model
                
                # Reset the model state for each evaluation
                evaluation_model.reset()
                
                result_ttt = evaluate_libero_task(task, eval_env, eval_obs, args, evaluation_model)

                results_ttt.append(result_ttt)
                
                print(f"Task {task_id} ({task_name}), Exp {exp_id}: {'SUCCESS' if result_ttt else 'FAILURE'}")

            # Update progress with current overall success rate
            current_success_rate = np.mean(results_ttt) * 100 if results_ttt else 0
            print(f"Completed Task {task_id} ({task_name}). Current overall success rate: {current_success_rate:.1f}%")
            
            # Clean up trained model for this task to free memory
            del trained_ttt_model
            del ttt_model
            del ttt_wrapped_model
            del ttt_trainer
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()

        def merge_multi_list(res):
            tmp = []
            for l in res:
                tmp.extend(l)
            return tmp

        def extract_iter_from_tqdm(tqdm_iter):
            return [_ for _ in tqdm_iter]

        eval_sequences = extract_iter_from_tqdm(eval_sequences)
        res_tup_ttt = [(res, eval_seq) for res, eval_seq in zip(results_ttt, eval_sequences)]
        all_res_tup_ttt = [copy.deepcopy(res_tup_ttt) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
        torch.distributed.gather_object(res_tup_ttt, all_res_tup_ttt, dst=0)

        if torch.distributed.get_rank() == 0:
            res_tup_list_ttt = merge_multi_list(all_res_tup_ttt)
            res_tup_list_ttt.sort(key=lambda x: x[1])
            print_and_save(res_tup_list_ttt, task_suite)

            all_results = [res[0] for res in res_tup_list_ttt]
            overall_success = np.mean(all_results) * 100
            print(f"\n=== LIBERO Test Time Training Final Results ===")
            print(f"Overall TTT Success Rate: {overall_success:.1f}%")
            print(f"Total evaluations: {len(all_results)}")
            print(f"Successful evaluations: {sum(all_results)}")
            print(f"Number of unique tasks trained: {len(task_groups)}")
            
            results_file = f"libero_ttt_results_{args.finetune_type}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'overall_success_rate': overall_success,
                    'total_evaluations': len(all_results),
                    'successful_evaluations': sum(all_results),
                    'unique_tasks_trained': len(task_groups),
                    'detailed_results': res_tup_list_ttt,
                    'args': vars(args)
                }, f, indent=2)
            print(f"Detailed results saved to {results_file}")

def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        print("this_result_list :", this_result_list)
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}:")
        print(f"{avg_success * 100:.1f}%")

def eval_one_epoch_libero_ddp(args, model, image_processor, tokenizer):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = UniAorldLiberoModelWrapper(
                        args,
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        libero_eval_max_steps=args.libero_eval_max_steps)
    evaluate_policy_ddp(args, wrapped_model, tokenizer, image_processor, cast_dtype)