<div align="center">   
  
# [CALVIN](https://github.com/mees/calvin)
</div>


## ⚙️ Installation

**(1) Install conda env**
```
conda create -n worldagen python=3.10
conda activate worldagen
```

**(2) Download CALVIN**
```
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
```

**(3) Download CALVIN ABC-D dataset**
```
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

**(4) Download third party packages**
```
cd ${YOUR_PATH_TO_WORLDAGEN}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**(Optional) Install OpenGL for headless server**
```python
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgles2-mesa libgles2-mesa-dev libgl1-mesa-glx libgl1-mesa-dev libosmesa6 libosmesa6-dev
```

**(5) Create a soft link to CALVIN**
```
cd ${YOUR_PATH_TO_WORLDAGEN}
ln -s $CALVIN_ROOT calvin
```

**(6) Copy the index file `except_lang_idx.npy` to the CALVIN ABC-D training data directory.**
```python
cp -r data_info/except_lang_idx/except_lang_idx.npy calvin/dataset/task_ABC_D/training
```

## Before Running

**(1) Download Relevant Checkpoints**

> For your convenience, we recommend using [gdown](https://github.com/wkentaro/gdown) to download checkpoints directly from Google Drive.

Download [MAE-Pretrained ViT-B Model](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=share_link). Make sure to place the downloaded checkpoint files in the appropriate directory `checkpoints/` (recommended path: `checkpoints/vit_mae/mae_pretrain_vit_base.pth`).

**(2) Update Config Files and Scripts**

The following variables should be updated to match your local paths and experiment naming (mainly in `scripts/calvin/*.sh`):

* **calvin_dataset_path**: the path to your CALVIN ABC-D dataset directory.
* **save_checkpoint_path**: the parent directory used to store experiment checkpoints. We recommend using `checkpoints/` under the project root.
* **resume_from_checkpoint**: the fine-tuned checkpoint path used for evaluation (typically determined by `experiment_name` + `ckpt_names`).

* **networkx:**
Due to compatibility issues between the networkx library in CALVIN and Python 3.10, we provide a compatible version of networkx.zip on the [website](https://drive.google.com/file/d/1z-d1SaI0rXfBtBicw1zPSsP-wE-26oLq/view?usp=sharing). Download and unzip it, then replace the existing networkx library.

## 🤖 Run WorldAgen

### Train (Calvin ABC-D)

```bash
bash scripts/calvin/train.sh
```

You will usually need to modify the following in `scripts/calvin/train.sh`:
- `calvin_dataset_path`
- `save_checkpoint_path`
- `vit_checkpoint_path`
- `--wandb_project`
- `--run_name` (experiment name)


### Evaluation

You can also download our pretrained [checkpoint](https://drive.google.com/file/d/1zubo1ckdc9Qlbnq4iNf9AD8V1oW8Z7Om/view?usp=sharing) and place it to the `checkpoints/scratch_qwen_16win_1img_5act` to run evaluation.

### Eval without Test Time Training
```bash
bash scripts/calvin/eval_wo_ttt.sh
```

You will need to modify the following in `scripts/calvin/eval_wo_ttt.sh`:
- `calvin_dataset_path` 
- `calvin_conf_path`
- `vit_checkpoint_path`
- `ckpt_names` (which checkpoints to evaluate, for example `"16"`)

### Eval with Test Time Training

```bash
bash scripts/calvin/eval_ttt.sh
```

In addition to the variables used for evaluation without TTT, TTT evaluation also requires adjusting:
- `lora_mode`: Whether to enable LoRA (Low-Rank Adaptation) fine-tuning. Typical values: `"lora"` (enable LoRA) or `"none"` (disable LoRA).
- `lora_rank`: The rank used in LoRA; controls the size of trainable parameters. 
- `lora_alpha`: The scaling factor for LoRA, controlling how much LoRA adapts the original weights.

- `ttt_num_samples`: Number of short rollouts sampled from each long trajectory during TTT. 
- `ttt_traj_len`: Length of each sampled short rollout from the trajectory.
- `ttt_sample_repeat`: Number of times to repeat rollout sampling per trajectory, enhancing diversity.

- `ttt_num_epoch`: Number of epochs to finetune each sample during TTT.

- `--ttt_data_dir`: Directory for TTT data. Customize this to match the location of your prepared TTT data.
