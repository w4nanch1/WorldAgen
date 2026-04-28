<div align="center">   
  
# [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
</div>


## ⚙️ Installation

**(1) Install conda env**
```
conda create -n worldagen python=3.10
conda activate worldagen
```

**(2) Install LIBERO**
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install transformers==4.40.2
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

**(3) Install WorldAgen dependencies**
```
cd ${YOUR_PATH_TO_WORLDAGEN}
pip install -r requirements.txt
```

**(Optional) Install OpenGL for headless server**
```python
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgles2-mesa libgles2-mesa-dev libgl1-mesa-glx libgl1-mesa-dev libosmesa6 libosmesa6-dev
```

## Before Running

**(1) Download Relevant Checkpoints**

> For convenience, we recommend using [gdown](https://github.com/wkentaro/gdown) to download checkpoints directly from Google Drive.

Download [MAE-Pretrained ViT-B Model](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=share_link). Place it under `checkpoints/` in this repository. The recommended path is `checkpoints/mae_pretrain_vit_base.pth`.

**(2) Update Config Files and Scripts**

The following variables should be updated to match your local paths and experiment naming (mainly in `scripts/libero/*.sh`):

* **save_dir / save_checkpoint_path**: the directory used to save checkpoints.
* **root_dir**: the root directory of your LIBERO-formatted dataset.
* **libero_path**: the path to your local LIBERO installation or benchmark assets used at runtime.
* **vit_checkpoint_path**: the path to the MAE-pretrained ViT checkpoint.
* **resume_from_checkpoint**: the checkpoint directory or checkpoint file used during evaluation.
* **LOG_DIR**: the directory used to save evaluation logs.

## 🤖 Run WorldAgen

### Train

```bash
bash scripts/libero/train.sh
```

You will usually need to modify the following in `scripts/libero/train.sh`:
- `save_dir`
- `root_dir`
- `vit_checkpoint_path`
- `libero_path`
- `calvin_dataset_path`
- `--wandb_project`
- `--run_name`

### Evaluation

You can also download our pretrained [checkpoint](https://drive.google.com/file/d/1IKRilL57qm52NEcHcLEL_ZFb2ak4sUMQ/view?usp=share_link) and place it to the `checkpoints/libero_traj7_len_1img_3act` to run evaluation. 

### Eval without Test Time Training

```bash
bash scripts/libero/eval_wo_ttt.sh
```

You will usually need to modify the following in `scripts/libero/eval_wo_ttt.sh`:
- `resume_from_checkpoint`
- `save_checkpoint_path`
- `vit_checkpoint_path`
- `pthlist` (which checkpoints to evaluate, for example `"38" "39"`)

### Eval with Test Time Training

```bash
bash scripts/libero/eval_ttt.sh
```

In addition to the variables used for evaluation without TTT, TTT evaluation also requires adjusting:
- `lora_mode`: whether to enable LoRA adaptation during TTT.
- `lora_rank`: the LoRA rank.
- `lora_alpha`: the LoRA scaling factor.
- `lora_dropout`: the LoRA dropout rate.
- `ttt_num_samples`: number of sampled segments used during TTT.
- `ttt_traj_len`: trajectory length of each sampled segment.
- `ttt_sample_repeat`: number of repeated samplings per evaluation trajectory.
- `ttt_batch_size`: batch size used during TTT optimization.
- `ttt_num_epoch`: number of optimization epochs during TTT.
- `ttt_learning_rate`: learning rate used during TTT.
- `ttt_weight_decay`: weight decay used during TTT.
- `ttt_data_dir`: directory containing the prepared TTT data.
