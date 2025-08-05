import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
import torch.nn.functional as F
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
from pdb import set_trace
import random 
from transformers import Qwen3Config
from models.qwen3 import Qwen3Model
import numpy as np

def generate_attention_mask(K, num_A, num_B, mask_l_obs_ratio, num_obs_token, resampler_query,
                            image_seq_len, action_seq_len, state_seq_len, action_pred_steps,
                             pred_obs_token, atten_goal, pred_state=False): 
    # num_A: 1+StateLength+ActionLength+self.NUM_RESAMPLER_QUERY*2*ImageLength+1*2*ImageLength
    # num_A: text, state, action, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN*ImageLength + StateLength(if pred state) + ActionLength * action_pred_steps
    # num_B: obs_tokens,  state_pred_token(if pred state), action_pred_token * action_pred_steps
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the pred part B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')

        # text + prev action + state + image -> action 
        if pred_state:
            attention_mask[start_index+num_A+pred_obs_token*image_seq_len+state_seq_len : end_index, start_index+1+state_seq_len : start_index+1+state_seq_len+action_seq_len] = -float('inf')
        else:
            attention_mask[start_index+num_A+pred_obs_token*image_seq_len : end_index, start_index+1+state_seq_len : start_index+1+state_seq_len+action_seq_len] = -float('inf')

        attention_mask[start_index : start_index+1+state_seq_len, start_index+1+state_seq_len : start_index+1+state_seq_len+action_seq_len] = -float('inf')
        attention_mask[start_index+1+state_seq_len+action_seq_len : start_index+num_A, start_index+1+state_seq_len : start_index+1+state_seq_len+action_seq_len] = -float('inf')

    for i in range(K):
        for j in range(K):
            start_x = i * (num_A + num_B)
            start_y = j * (num_A + num_B)
            end_x = start_x + num_A + num_B
            end_y = start_y + num_A + num_B
            # state and img can not attend to text
            attention_mask[start_x : start_x+1, start_y+1 : start_y+1+state_seq_len] = -float('inf')
            attention_mask[start_x : start_x+1, start_y+1+state_seq_len+action_seq_len : start_y+num_A] = -float('inf')

            attention_mask[start_x+1 : start_x+1+state_seq_len, start_y : start_y+1] = -float('inf')
            attention_mask[start_x+1+state_seq_len+action_seq_len : start_x+num_A, start_y : start_y+1] = -float('inf')
            # action + state + image -> image, state
            if atten_goal:
                if pred_state:
                    attention_mask[start_x+num_A:start_x+num_A+pred_obs_token*image_seq_len+state_seq_len] = -float('inf')
                    attention_mask[start_x+num_A:start_x+num_A+pred_obs_token*image_seq_len+state_seq_len, start_x+1: start_x+num_A] = 0.0

                    attention_mask[start_x+num_A+pred_obs_token*image_seq_len+state_seq_len : end_x, start_y : start_y+1] = -float('inf')
                else:
                    attention_mask[start_x+num_A:start_x+num_A+pred_obs_token*image_seq_len] = -float('inf')
                    attention_mask[start_x+num_A:start_x+num_A+pred_obs_token*image_seq_len, start_x+1: start_x+num_A] = 0.0

                    attention_mask[start_x+num_A+pred_obs_token*image_seq_len : end_x, start_y : start_y+1] = -float('inf')
                    
            if pred_state:
                attention_mask[start_x+num_A:start_x+num_A+pred_obs_token*image_seq_len+state_seq_len, start_y : start_y+1] = -float('inf')
            else:
                if pred_obs_token == 0:
                    pass
                else:
                    attention_mask[start_x+num_A : start_x+num_A+pred_obs_token*image_seq_len, start_y : start_y+1] = -float('inf')
        
    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

class UniAorld(nn.Module):
    def __init__(
        self,
        clip_device,
        vit_checkpoint_path,
        sequence_length=10,
        image_sequence_length=5,
        state_sequence_length=5,
        action_sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=9,
        mask_l_obs_ratio=0.0,
        calvin_input_image_size=224,
        patch_size=16,
        transformer_layers=12,
        hidden_dim=384,
        transformer_heads=12,
        phase="",
        gripper_width=False,
        pred_state=False,
        action_pred_steps=1,
        pred_image=True,
        atten_goal=0,
        use_qwen=False,
    ):
        super().__init__()
        self.device = clip_device
        self.sequence_length = sequence_length
        self.mask_l_obs_ratio = mask_l_obs_ratio
        self.hidden_dim = hidden_dim
        self.phase = phase
        self.use_qwen = use_qwen
        self.pred_image = pred_image
        assert self.phase in ["pretrain", "finetune", "evaluate"]
        self.gripper_width = gripper_width
        self.vit_checkpoint_path = vit_checkpoint_path
        self.image_sequence_length = image_sequence_length
        self.action_sequence_length = action_sequence_length
        self.state_sequence_length = state_sequence_length
        self.pred_state = pred_state
        self.action_pred_steps = action_pred_steps
        self.atten_goal = atten_goal

        # text projector
        self.text_projector = nn.Linear(512, self.hidden_dim)        

        # state encoder
        ARM_STATE_FEATURE_DIM = self.hidden_dim 
        GRIPPER_STATE_FEATURE_DIM = self.hidden_dim
        self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.state_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        # action encoder
        self.action_pose_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.action_gripper_position_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.action_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        # vision encoder (frozen)
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # resampler
        self.RESAMPLER_hidden_dim = 768  
        self.NUM_RESAMPLER_QUERY = num_resampler_query
        self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_hidden_dim, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
        self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_primary_projector = nn.Linear(768, self.hidden_dim)
        self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_wrist_projector = nn.Linear(768, self.hidden_dim)

        # action_pred_token
        if self.action_pred_steps > 0:
            self.action_pred_token = nn.Parameter(torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim))

        # state pred token
        if self.pred_state:
            self.state_pred_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))

        # obs_token
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2
        self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
        
        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)

        # num_non_learnable_token_per_timestep = 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim), requires_grad=True)
        if not use_qwen:
            config = GPT2Config()
            config.hidden_size = self.hidden_dim
            config.n_layer = transformer_layers
            config.vocab_size = 1
            config.n_head = transformer_heads
            self.transformer_backbone = GPT2Model(config)
        else:
            config = Qwen3Config()
            config.hidden_size = self.hidden_dim
            config.num_hidden_layers = transformer_layers 
            config.vocab_size = 1
            config.num_attention_heads = transformer_heads 
            config.num_key_value_heads = transformer_heads
            config.intermediate_size = 4 * self.hidden_dim 
            self.transformer_backbone = Qwen3Model(config)

        # action decoder
        MLP_hidden_dim = self.hidden_dim // 2
        self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.GELU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.GELU(),
            nn.Linear(MLP_hidden_dim, self.hidden_dim),
        )
        self.arm_action_decoder = nn.Linear(self.hidden_dim, 6)
        self.gripper_action_decoder = nn.Linear(self.hidden_dim, 1)

        # state decoder
        if self.pred_state:
            self.state_decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, MLP_hidden_dim),
                nn.GELU(),
                nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
                nn.GELU(),
                nn.Linear(MLP_hidden_dim, self.hidden_dim),
            )
            self.arm_state_decoder = nn.Linear(self.hidden_dim, 6)
            self.gripper_state_decoder = nn.Linear(self.hidden_dim, 1)

        # image decoder
        if self.pred_image:
            self.IMAGE_DECODER_hidden_dim = 512
            self.NUM_MASK_TOKEN = 196
            self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim))
            self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim))
            self.image_decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.IMAGE_DECODER_hidden_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=6,
            )
            self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
            self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, 16 * 16 * 3)

        self.initialize_weights()
        self._init_model_type()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_model_type(self):
        self.vision_encoder_type = next(self.vision_encoder.parameters()).dtype
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).dtype
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).dtype

    def forward(self, image_primary, image_wrist, state, text_token, action):
        B, S_IMAGE, S_STATE, S_ACTION = image_primary.shape[0], self.image_sequence_length, self.state_sequence_length, self.action_sequence_length
        num_chunk = self.sequence_length
        S_IMAGE = S_IMAGE * num_chunk
        S_STATE = S_STATE * num_chunk
        S_ACTION = S_ACTION * num_chunk

        # attention mask
        self.attention_mask = generate_attention_mask(
            K=1, num_A=1+S_STATE+S_ACTION+self.NUM_RESAMPLER_QUERY*2*S_IMAGE+1*2*S_IMAGE, 
            num_B=self.NUM_OBS_TOKEN*S_IMAGE + S_STATE*(self.pred_state) + S_ACTION*self.action_pred_steps,
            mask_l_obs_ratio=self.mask_l_obs_ratio, num_obs_token=self.NUM_OBS_TOKEN, 
            resampler_query=self.NUM_RESAMPLER_QUERY, image_seq_len=S_IMAGE, action_seq_len=S_ACTION, 
            state_seq_len=S_STATE, action_pred_steps=self.action_pred_steps, pred_obs_token=self.pred_image, 
            atten_goal=self.atten_goal, pred_state=self.pred_state
        )

        # text feature
        text_embedding = self.text_projector(text_token)
        text_embedding = text_embedding.view(B, num_chunk, -1, self.hidden_dim)

        # action feature
        if action.type() != self.transformer_backbone_type:
            action = action.type(self.transformer_backbone_type)
        arm_action_feature = self.action_pose_encoder(action[:, :, :, :6])
        if self.gripper_width:
            gripper_action_one_hot = F.one_hot(action[:, :, :, 6:].long(), num_classes=2).float()
            gripper_action_feature = self.action_gripper_position_encoder(gripper_action_one_hot.type_as(action))
        else:
            gripper_action_feature = self.action_gripper_position_encoder(action[:, :, :, 6:])
        action_embedding = self.action_projector(torch.cat((arm_action_feature, gripper_action_feature), dim=3))
        action_embedding = action_embedding.view(B, num_chunk, -1, self.hidden_dim)

        # state feature
        if state.type() != self.transformer_backbone_type:
            state = state.type(self.transformer_backbone_type)
        arm_state_feature = self.arm_state_encoder(state[:, :, :, :6])
        if self.gripper_width:
            gripper_state_one_hot = F.one_hot(state[:, :, :, 6:].long(), num_classes=2).float()
            gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(state))
        else:
            gripper_state_feature = self.gripper_state_encoder(state[:, :, :, 6:])
        
        state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=3))
        state_embedding = state_embedding.view(B, num_chunk, -1, self.hidden_dim) 

        # image feature 
        if image_primary.type() != self.vision_encoder_type:
            image_primary = image_primary.type(self.vision_encoder_type)
            image_wrist = image_wrist.type(self.vision_encoder_type)
        with torch.no_grad():
            image_primary_feature, _, _ = self.vision_encoder.forward_encoder(image_primary.flatten(0, 1), mask_ratio=0.0)
            image_wrist_feature, _, _ = self.vision_encoder.forward_encoder(image_wrist.flatten(0, 1), mask_ratio=0.0)

        if image_primary_feature.type() != self.perceiver_resampler_type:
            image_primary_feature = image_primary_feature.type(self.perceiver_resampler_type)
            image_wrist_feature = image_wrist_feature.type(self.perceiver_resampler_type)
        image_primary_feature = image_primary_feature.view(B, S_IMAGE, image_primary_feature.shape[-2], image_primary_feature.shape[-1])
        image_wrist_feature = image_wrist_feature.view(B, S_IMAGE, image_wrist_feature.shape[-2], image_wrist_feature.shape[-1])
        image_primary_cls_token = image_primary_feature[:, :, :1, :]
        image_wrist_cls_token = image_wrist_feature[:, :, :1, :]
        image_primary_feature = image_primary_feature[:, :, 1:, :]
        image_wrist_feature = image_wrist_feature[:, :, 1:, :]

        # perceiver resampler
        image_primary_feature = self.perceiver_resampler(image_primary_feature.reshape(B*S_IMAGE, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
        image_wrist_feature = self.perceiver_resampler(image_wrist_feature.reshape(B*S_IMAGE, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
        image_primary_embedding = self.image_primary_projector(image_primary_feature.flatten(0, 2)).view(B, S_IMAGE, -1, self.hidden_dim)
        image_wrist_embedding = self.image_wrist_projector(image_wrist_feature.flatten(0, 2)).view(B, S_IMAGE, -1, self.hidden_dim)
        image_embedding = torch.cat((image_primary_embedding, image_wrist_embedding), dim=2).view(B, num_chunk, -1, self.hidden_dim)
        image_cls_token_primary_embedding = self.cls_token_primary_projector(image_primary_cls_token.flatten(0, 2)).view(B, S_IMAGE, -1, self.hidden_dim)
        image_cls_token_wrist_embedding = self.cls_token_wrist_projector(image_wrist_cls_token.flatten(0, 2)).view(B, S_IMAGE, -1, self.hidden_dim)
        image_cls_token_embedding = torch.cat((image_cls_token_primary_embedding, image_cls_token_wrist_embedding), dim=2).view(B, num_chunk, -1, self.hidden_dim)

        # concat multi-modality data
        embeddings = torch.cat((text_embedding, state_embedding, action_embedding, image_embedding, image_cls_token_embedding), dim=2)
        pred_token_start_idx = embeddings.shape[2]
        transformer_input_list = [embeddings]
        if self.pred_image:
            transformer_input_list.append(self.obs_tokens.repeat(B, S_IMAGE, 1, 1).view(B, num_chunk, -1, self.hidden_dim))
        if self.pred_state:
            transformer_input_list.append(self.state_pred_token.repeat(B, S_IMAGE, 1, 1).view(B, num_chunk, -1, self.hidden_dim))

        transformer_input_list.append(self.action_pred_token.repeat(B, S_ACTION, 1, 1).view(B, num_chunk, -1, self.hidden_dim))

        transformer_input = torch.cat(transformer_input_list, dim=2)  
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(B, num_chunk, transformer_input.shape[-2], 1)
        transformer_input = transformer_input.flatten(1, 2)
        # causal transformer forward
        if transformer_input.type() != self.transformer_backbone_type:
            transformer_input = transformer_input.type(self.transformer_backbone_type)
        transformer_input = self.embedding_layer_norm(transformer_input) 
        if self.use_qwen:
            transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)).last_hidden_state
        else:
            transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)
        transformer_output = transformer_output.view(B, num_chunk, -1, self.hidden_dim)
        image_pred = None
        this_num_obs_token = 0
        if self.pred_image:
            this_num_obs_token = self.NUM_OBS_TOKEN
            obs_pred_feature = transformer_output[:, :, pred_token_start_idx:pred_token_start_idx+self.NUM_OBS_TOKEN*self.image_sequence_length, :]
            obs_pred_embedding = self.image_decoder_obs_pred_projector(obs_pred_feature.reshape(-1, self.hidden_dim))
            obs_pred_embedding = obs_pred_embedding.view(B * S_IMAGE * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_OBS_TOKEN_PER_IMAGE, self.IMAGE_DECODER_hidden_dim)
            mask_tokens = self.mask_token.repeat(B * S_IMAGE * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1)
            image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1) 
            image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
            image_decoder_output = self.image_decoder(image_decoder_input) 
            image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :]
            image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, self.IMAGE_DECODER_hidden_dim))
            image_pred = self.image_decoder_pred(image_pred_feature)  
            image_pred = image_pred.view(B * S_IMAGE, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, -1)  

        if self.pred_state:
            state_pred_feature = transformer_output[:, :, pred_token_start_idx+this_num_obs_token*self.image_sequence_length\
                                                    :pred_token_start_idx+this_num_obs_token*self.image_sequence_length+self.state_sequence_length, :]
            state_pred_feature = self.state_decoder(state_pred_feature)
            arm_pred_state = self.arm_state_decoder(state_pred_feature)
            gripper_pred_state = self.gripper_state_decoder(state_pred_feature)

        if self.pred_state:
            action_pred_feature = transformer_output[:, :, pred_token_start_idx+this_num_obs_token*self.image_sequence_length+self.state_sequence_length\
                                                 :pred_token_start_idx+this_num_obs_token*self.image_sequence_length+self.state_sequence_length+self.action_sequence_length*self.action_pred_steps, :]
        else:
            action_pred_feature = transformer_output[:, :, pred_token_start_idx+this_num_obs_token*self.image_sequence_length
                                                 :pred_token_start_idx+this_num_obs_token*self.image_sequence_length+self.action_sequence_length*self.action_pred_steps, :]
        action_pred_feature = self.action_decoder(action_pred_feature)
        arm_pred_action = self.arm_action_decoder(action_pred_feature)
        gripper_pred_action = self.gripper_action_decoder(action_pred_feature)
        
        return arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state
    
