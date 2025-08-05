import torch 
import torch.nn as nn
import loralib as lora
from peft import LoraConfig, get_peft_model

class LoraModel(nn.Module):
    def __init__(self, model, lora_rank=4, lora_alpha=8, lora_dropout=0.1, lora_mode="lora", backbone='qwen3'):
        super(LoraModel, self).__init__()
        assert lora_mode in ["lora", "fft"]
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

        elif self.lora_mode == "fft":
            for param in self.model.module.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Invalid mode: {self.lora_mode}")
        return self.model

    def _get_trainable_parameters(self):
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

        elif self.lora_mode == "fft":
            trainable_params = []
            total_trainable_count = 0
            trainable_param_names = []
            
            for name, param in self.model.module.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
                trainable_param_names.append(f"net.{name}")
                total_trainable_count += param.numel()
            
            self.trainable_param_names = trainable_param_names
            print(f"Full-finetune All Network: train {len(trainable_param_names)} tensor in network, including {total_trainable_count} parameters totally")
            return trainable_params
        
        else:
            raise ValueError(f"Invalid mode: {self.lora_mode}")