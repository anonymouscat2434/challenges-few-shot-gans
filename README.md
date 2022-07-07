# Overcoming challenges in leveraging GANs for few-shot data augmentation

Authors: anonymous

In this paper, we explore the use of GAN-based few-shot data augmentation as a method to improve few-shot classification performance. We perform an exploration into how a GAN can be fine-tuned for such a task (one of which is in a \emph{class-incremental} manner), as well as a rigorous empirical investigation into how well these models can perform to improve few-shot classification. We identify issues related to the difficulty of training such generative models under a purely supervised regime with very few examples, as well as issues regarding the evaluation protocols of existing works. We also find that in this regime, classification accuracy is highly sensitive to how the classes of the dataset are randomly split. Therefore, we propose a semi-supervised fine-tuning approach as a more pragmatic way forward to address these problems, and demonstrate gains in FID and precision-recall metrics as well as classification performance.

# Installation

## Downloading pre-trained checkpoints

After cloning the repository, download the pre-trained checkpoints. You can download them here at this [anonymised link](https://mega.nz/file/oeYBVLpY#U_ibaV-dj4mvHO7NXOzzOlB_1vKCWA1_rUOfzUQJ8uE). (NOTE: mega.nz is being used here for the purposes of this anonymised repository, and the weights will be hosted on Google Drive post-rebuttal.)

Once you have `fsi-backup.tar.gz` downloaded, untar it in some directory of your choosing, e.g:

```
cd /somedir/checkpoints
tar -xvzf fsi-backup.tar.gz
```

## Updating paths

A few things must be set up in env.sh:
- The `results` variable must point to whatever directory you extracted the pre-trained checkpoints to. For instance,
  if you un-tared `fsi-backup.tar.gz` into `/somedir/checkpoints`, then you should have `export results=/somedir/checkpoints`.
- You must have the Git repository for StyleGAN2-ADA in your `PYTHONPATH`. Simply git clone [this repo](https://github.com/NVlabs/stylegan2-ada-pytorch)
  somewhere and export `STYLEGAN2_ADA_PYTORCH` to this directory. `env.sh` will automatically append this to your `PYTHONPATH`.
  
When both these steps are done, source the `env.sh` file like so:
  
```
source exps/env.sh
```

## Dependencies

```
tqdm
torch==1.7.1
torchvision==0.8.2
pytorch-fid==0.2.1
Pillow==8.3.2
pandas==1.4.2
prdc==0.2
```

`all_requirements.txt` are all the packages that exist in my internal development environment, but I do not recommend installing from this since there will be a lot of dependencies not related to this project specifically.

# Training

## Reproducing an experiment

`env.sh` specifies an environment variable for each experiment. If we look at this file, we can see that each experiment's path is
specified by a particular environment variable. For instance, `EMNIST_STAGE2A_S0` corresponds to the GAN pre-training experiment
for seed=0, while `EMNIST_STAGE2B_K5_S0` is the GAN fine-tuning experiment for `seed=0` when `k=5`. The corresponding semi-supervised
experiment is called `EMNIST_STAGE2B_K5_SEMI_ALPHA_S0`. If we want to reproduce the experiment for `EMNIST_STAGE2A_S0`, we can simply
invoke the launch script like so (after sourcing `exps/env.sh`):

```
WORLD_SIZE=0 python launch.py \
--json_file=$EMNIST_STAGE2A_S0/exp_dict.json \
--tl=trainval.py \
--savedir=/tmp/emnist-stage2a-seed0 \
-d /mnt/public/datasets
```

Here, `-d <data dir>` specifies where the data is located, which in this case is where the EMNIST dataset will be downloaded to. `--savedir` specifies the directory where the experiment should be saved.

## Grokking metrics

Note that each pre-trained experiment directory contains a `score_list.pkl` file, which is the raw training metrics file for when we trained that experiment internally. These can be easily viewed with the following code:

```
import pickle
import os
pkl_file = os.environ['EMNIST_STAGE2A_S0'] + "/score_list.pkl"
pkl = pickle.load(open(pkl_file, "rb"))
pandas_df = pd.DataFrame(pkl)
```

Alternatively, one can use `pkl2csv.py` to convert it to csv:

```
python pkl2csv.py $EMNIST_STAGE2A_S0/score_list.pkl score_list.csv
```

## Loading pre-trained checkpoints

Each experiment contains usually two `.pth` checkpoint files. `model.pth` is the checkpoint corresponding to the very last epoch the experiment was trained for, while `model.<metric>.pth` is the checkpoint corresponding to the best metric. For instance, for `*STAGE2A` (GAN pre-training) experiments this is `fid`, so you will see `model.fid.pth`. For `*STAGEB` this is usually `model.valid_fid.pth`.

To output generated images:

```
python eval.py --experiment=$EMNIST_STAGE2A_S0
```

# Miscellaneous

## Exporting datasets

If you would prefer to simply use the exact same dataset splits we used without running our code, this is also possible. The following [Google Drive link](https://mega.nz/folder/gWRlSKrb#AErTo0pwQv2y3KW0hkTLAA)  provides access to each dataset used as well as their splits. Each pkl has the naming convention `{dataset}-s{seed}-{res}px-k{kshot}-{split}.pkl`. For example, these files correspond to EMNIST with `seed=0`, 32px resolution, and `k_shot=5`:

```
emnist_fs-s0-32px-k5-train.pkl
emnist_fs-s0-32px-k5-supports.pkl
emnist_fs-s0-32px-k5-valid.pkl
emnist_fs-s0-32px-k5-test.pkl      
```
