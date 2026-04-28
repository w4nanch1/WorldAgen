import os
import sys
import unittest.mock as mock

def headless_environment():
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'

    class MockGL:
        def __init__(self):
            self.glGetError = mock.MagicMock(return_value=0)
            self.GL_NO_ERROR = 0
            
        def __getattr__(self, name):
            return mock.MagicMock()
    
    class MockPlatform:
        def __init__(self):
            self.GL = MockGL()
            
        def __getattr__(self, name):
            return mock.MagicMock()
    
    class MockErrorChecker:
        def __init__(self, *args, **kwargs):
            pass
            
        def __call__(self, *args, **kwargs):
            return None
    
    mock_gl = MockGL()
    mock_platform = MockPlatform()
    
    opengl_modules = {
        'OpenGL': mock.MagicMock(),
        'OpenGL.platform': mock_platform,
        'OpenGL.platform.osmesa': mock_platform,
        'OpenGL.GL': mock_gl,
        'OpenGL.GL.VERSION': mock.MagicMock(),
        'OpenGL.GL.VERSION.GL_1_1': mock.MagicMock(),
        'OpenGL.raw': mock.MagicMock(),
        'OpenGL.raw.GL': mock.MagicMock(),
        'OpenGL.raw.GL.VERSION': mock.MagicMock(),
        'OpenGL.raw.GL.VERSION.GL_1_1': mock.MagicMock(),
        'OpenGL.arrays': mock.MagicMock(),
        'OpenGL.error': mock.MagicMock(),
    }
    
    errors_module = mock.MagicMock()
    errors_module._ErrorChecker = MockErrorChecker
    opengl_modules['OpenGL.raw.GL._errors'] = errors_module
    
    for name, module in opengl_modules.items():
        sys.modules[name] = module
    
    sys.modules['OpenGL'].GL = mock_gl
    sys.modules['OpenGL'].platform = mock_platform
    
    print(f"[Process {os.getpid()}] OpenGL completely mocked for headless environment")

# headless_environment()

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
import random
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed_utils import init_distributed_device, world_info_from_env

from utils.eval_utils_libero import eval_one_epoch_libero_ddp

# try:
#     from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
# except:
#     pass 
# from utils.eval_utils_libero import eval_one_epoch_libero_ddp as eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record
# from utils.arguments_utils import get_args_and_cfg
from utils.arguments_utils import get_parser
from pdb import set_trace
from models.seer_model import SeerAgent
from models.model import UniAorld

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)

    # model
    model = UniAorld(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        )

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

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

    if args.finetune_type == "libero_10":
        eval_one_epoch_libero_ddp(
            args=args,
            model=ddp_model,
            image_processor=model.image_processor,
            tokenizer=clip,
        )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()