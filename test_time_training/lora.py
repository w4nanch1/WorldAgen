import torch 
import torch.nn as nn
import loralib as lora
import types
import math 

from peft import LoraConfig, get_peft_model

class LoRALayer(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Get dimensions from the base layer
        if isinstance(base_layer, nn.Linear):
            in_dim, out_dim = base_layer.in_features, base_layer.out_features
        else:
            # Handle other layer types if needed
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")
        
        # LoRA components
        self.lora_down = nn.Linear(in_dim, r, bias=False)
        self.lora_up = nn.Linear(r, out_dim, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        self.scaling = lora_alpha / r
        
        # Initialize with small weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    def forward(self, x):
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
     
class LoraModel(nn.Module):
    def __init__(self, model, lora_rank=4, lora_alpha=8, lora_dropout=0.1, lora_mode="lora", backbone='qwen3'):
        super(LoraModel, self).__init__()
        assert lora_mode in ["lora", "ft_backbone", "fft", "img_pred_head", "ft_backbone_and_img_pred", "vision_encoder", "vision_encoder_lora"]
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_param_names = []
        self.backbone = backbone
        self.lora_mode = lora_mode
        
    def _apply_finetuning_to_gpt2(self):
        target_modules=["c_attn", "c_proj", "c_fc"] if self.backbone=='gpt2' \
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                # else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        if self.lora_mode == "lora":
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules, 
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            backbone = self.model.module.transformer_backbone
            peft_backbone = get_peft_model(backbone, lora_config)
            self.model.module.transformer_backbone = peft_backbone
            self.model.module.attention_mask.requires_grad_(False)
            
        elif self.lora_mode == "ft_backbone":
            # Enable gradient for the entire backbone regardless of LoRA
            for param in self.model.module.transformer_backbone.parameters():
                param.requires_grad = True
        elif self.lora_mode == "img_pred_head":
            # Freeze all parameters first
            for param in self.model.module.parameters():
                param.requires_grad = False
                
            # Enable gradient only for image prediction components
            for param in self.model.module.image_decoder_obs_pred_projector.parameters():
                param.requires_grad = True
            
            for param in self.model.module.image_decoder.parameters():
                param.requires_grad = True
                
            for param in self.model.module.image_decoder_norm.parameters():
                param.requires_grad = True
                
            for param in self.model.module.image_decoder_pred.parameters():
                param.requires_grad = True
            
            # Enable learnable embeddings related to image decoding
            if hasattr(self.model.module, 'mask_token') and hasattr(self.model.module.mask_token, 'requires_grad'):
                self.model.module.mask_token.requires_grad = True
                
            if hasattr(self.model.module, 'image_decoder_position_embedding'):
                self.model.module.image_decoder_position_embedding.requires_grad = True

        elif self.lora_mode == "ft_backbone_and_img_pred":
            # First freeze all parameters
            for param in self.model.module.parameters():
                param.requires_grad = False
            
            # Enable gradient for the backbone
            for param in self.model.module.transformer_backbone.parameters():
                param.requires_grad = True
            
            # Enable gradient for image prediction components
            for param in self.model.module.image_decoder_obs_pred_projector.parameters():
                param.requires_grad = True
            
            for param in self.model.module.image_decoder.parameters():
                param.requires_grad = True
                
            for param in self.model.module.image_decoder_norm.parameters():
                param.requires_grad = True
                
            for param in self.model.module.image_decoder_pred.parameters():
                param.requires_grad = True
            
            # Enable learnable embeddings related to image decoding
            if hasattr(self.model.module, 'mask_token') and hasattr(self.model.module.mask_token, 'requires_grad'):
                self.model.module.mask_token.requires_grad = True
                
            if hasattr(self.model.module, 'image_decoder_position_embedding'):
                self.model.module.image_decoder_position_embedding.requires_grad = True

        elif self.lora_mode == "vision_encoder":
            # Freeze all parameters first
            for param in self.model.module.parameters():
                param.requires_grad = False
                
            # Enable gradient only for vision encoder parameters
            for param in self.model.module.vision_encoder.parameters():
                param.requires_grad = True

        elif self.lora_mode == "vision_encoder_lora":
            # First, freeze all parameters
            for param in self.model.module.parameters():
                param.requires_grad = False
            
            # Manually add LoRA layers to the vision encoder attention layers
            # We need to manually identify the attention layers in the vision encoder
            vision_encoder = self.model.module.vision_encoder
            
            # Add LoRA to the encoder attention blocks
            for i, block in enumerate(vision_encoder.blocks):
                # Replace the query, key, value projection with LoRA
                orig_qkv = block.attn.qkv
                orig_proj = block.attn.proj
                
                # Create and attach LoRA layers
                block.attn.qkv_lora = LoRALayer(
                    orig_qkv, 
                    r=self.lora_rank, 
                    lora_alpha=self.lora_alpha, 
                    lora_dropout=self.lora_dropout
                )
                block.attn.proj_lora = LoRALayer(
                    orig_proj, 
                    r=self.lora_rank, 
                    lora_alpha=self.lora_alpha, 
                    lora_dropout=self.lora_dropout
                )
                
                # Modify the forward pass to use LoRA
                def new_forward_qkv(self, x):
                    return self.qkv(x) + self.qkv_lora(x)
                    
                def new_forward_proj(self, x):
                    return self.proj(x) + self.proj_lora(x)
                
                # Monkey patch the forward methods
                block.attn.forward_qkv = types.MethodType(new_forward_qkv, block.attn)
                block.attn.forward_proj = types.MethodType(new_forward_proj, block.attn)

        else:
            # Enable gradient for the entire network regardless of LoRA
            for param in self.model.module.parameters():
                param.requires_grad = True

        return self.model

    def _get_trainable_parameters(self):
        # First, disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        if self.lora_mode == "lora":
            lora_params = []  
            trainable_params = 0
            lora_param_names = [] 
            
            for name, param in self.model.named_parameters():
                if 'lora_' in name:  
                    param.requires_grad = True
                    lora_params.append(param) 
                    lora_param_names.append(name)
                    trainable_params += param.numel()
            
            self.lora_param_names = lora_param_names  
            print(f"LoRA mode: train {len(lora_param_names)} lora tensor, including {trainable_params} parameters totally")
            return lora_params
        
        elif self.lora_mode == "ft_backbone":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            for name, param in self.model.module.transformer_backbone.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                trainable_param_names.append(f"transformer_backbone.{name}")
                total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"FT GPT2 Transformer Backbone: train {len(trainable_param_names)} tensor in transformer backbone, including {total_trainable_count} parameters totally")
            return trainable_params
        
        elif self.lora_mode == "img_pred_head":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            # Enable gradients only for image prediction components
            image_component_prefixes = [
                "image_decoder_obs_pred_projector",
                "image_decoder",
                "image_decoder_norm",
                "image_decoder_pred",
                "mask_token",
                "image_decoder_position_embedding"
            ]
            
            for name, param in self.model.module.named_parameters():
                # Check if parameter is part of image prediction components
                if any(name.startswith(prefix) for prefix in image_component_prefixes):
                    param.requires_grad = True
                    trainable_params.append(param)
                    trainable_param_names.append(f"net.{name}")
                    total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"Image Prediction Head Finetuning: train {len(trainable_param_names)} tensor in image prediction components, including {total_trainable_count} parameters totally")
            return trainable_params
        
        elif self.lora_mode == "vision_encoder":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            for name, param in self.model.module.vision_encoder.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                trainable_param_names.append(f"vision_encoder.{name}")
                total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"Vision Encoder Finetuning: train {len(trainable_param_names)} tensor in vision encoder, including {total_trainable_count} parameters totally")
            return trainable_params

        elif self.lora_mode == "ft_backbone_and_img_pred":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            # Enable gradients for backbone
            for name, param in self.model.module.transformer_backbone.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                trainable_param_names.append(f"transformer_backbone.{name}")
                total_trainable_count += param.numel()
                
            # Enable gradients for image prediction components
            image_component_prefixes = [
                "image_decoder_obs_pred_projector",
                "image_decoder",
                "image_decoder_norm",
                "image_decoder_pred",
                "mask_token",
                "image_decoder_position_embedding"
            ]
            
            for name, param in self.model.module.named_parameters():
                # Check if parameter is part of image prediction components
                if any(name.startswith(prefix) for prefix in image_component_prefixes):
                    param.requires_grad = True
                    trainable_params.append(param)
                    trainable_param_names.append(f"net.{name}")
                    total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"Finetuning Backbone and Image Prediction Head: train {len(trainable_param_names)} tensors, including {total_trainable_count} parameters totally")
            return trainable_params

        elif self.lora_mode == "vision_encoder_lora":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            # Collect LoRA parameters from vision encoder
            vision_encoder = self.model.module.vision_encoder
            for i, block in enumerate(vision_encoder.blocks):
                if hasattr(block.attn, 'qkv_lora'):
                    for name, param in block.attn.qkv_lora.named_parameters():
                        param.requires_grad = True
                        trainable_params.append(param)
                        trainable_param_names.append(f"vision_encoder.blocks.{i}.attn.qkv_lora.{name}")
                        total_trainable_count += param.numel()
                        
                if hasattr(block.attn, 'proj_lora'):
                    for name, param in block.attn.proj_lora.named_parameters():
                        param.requires_grad = True
                        trainable_params.append(param)
                        trainable_param_names.append(f"vision_encoder.blocks.{i}.attn.proj_lora.{name}")
                        total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"Vision Encoder LoRA mode: train {len(trainable_param_names)} lora tensors, including {total_trainable_count} parameters totally")
            return trainable_params

        else:
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            for name, param in self.model.module.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                trainable_param_names.append(f"net.{name}")
                total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"FFT All Network: train {len(trainable_param_names)} tensor in network, including {total_trainable_count} parameters totally")
            return trainable_params