# MaskDiME's official code

This is the codebase for the CVPR 2026 paper [MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations]([https://arxiv.org/abs/2203.15636](https://arxiv.org/abs/2602.18792)).


## Environment

Through anaconda, install our environment:

```bash
conda env create -f env.yaml
conda activate maskdime
``` 

## Data preparation

Please download and uncompress the CelebA dataset [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). There is no need for any post-processing. The final folder structure should be:

```
PATH ---- img_align_celeba ---- xxxxxx.jpg
      |
      --- list_attr_celeba.csv
      |
      --- list_eval_partition.csv
```

## Downloading pre-trained models

To use our trained models, you must download them first from this [link](https://huggingface.co/guillaumejs2403/DiME). Please extract them to the folder `models`. We provides the CelebA diffusion model, the classifier under observation, and the trained oracle. Finally, download the VGGFace2 model throught this [github repo](https://github.com/cydonia999/VGGFace2-pytorch). Download the `resnet50_ft` model.

## Extracting Counterfactual Explanations

To create the counterfactual explanations, please use the main.py script as follows:

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 \
             --class_cond False \
             --diffusion_steps 500 \
             --learn_sigma True \
             --noise_schedule linear \
             --num_channels 128 \
             --num_heads 4 \
             --num_res_blocks 2 \
             --resblock_updown True \
             --use_fp16 True \
             --use_scale_shift_norm True"



SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"

DATAPATH="dataset/CelebA"
OUTPUT_PATH="CelebA/MaskDiME/smile/"
MODELPATH="DiME/models/ddpm-celeba.pt"
CLASSIFIERPATH="DiME/models/classifier.pth"
ORACLEPATH="DiME/models/oracle.pth"
EXPNAME="exp/name"

GPU=0
S=60
SEED=4
NUMBATCHES=9999
USE_LOGITS=True
CLASS_SCALES="8,10,15"
LAYER=18
PERC=30
L1=0.05
QUERYLABEL=31
TARGETLABEL=-1
IMAGESIZE=128
CLIP_SIGMA=3
BLUR_KERNEL=5
BLUR_SIGMA=3
GRADSCALE=8
TOPKRATIO=0.05
XTRATE=0.5



python -W ignore maskdime4celeba.py \
  $MODEL_FLAGS $SAMPLE_FLAGS \
  --query_label $QUERYLABEL \
  --target_label $TARGETLABEL \
  --output_path "$OUTPUT_PATH" \
  --num_batches $NUMBATCHES \
  --start_step $S \
  --dataset 'CelebA' \
  --data_dir $DATAPATH \
  --exp_name "$EXPNAME" \
  --gpu $GPU \
  --model_path "$MODELPATH" \
  --classifier_scales "$CLASS_SCALES" \
  --classifier_path "$CLASSIFIERPATH" \
  --seed $SEED \
  --oracle_path "$ORACLEPATH" \
  --l1_loss $L1 \
  --use_logits $USE_LOGITS \
  --l_perc $PERC \
  --l_perc_layer $LAYER \
  --save_x_t False \
  --save_z_t False \
  --save_images True \
  --image_size $IMAGESIZE \
  --grad_scale $GRADSCALE \
  --topk_ratio  $TOPKRATIO \
  --mask True \
  --method "maskdime" \
  --grads_clip False \
  --x_t_rate $XTRATE \
  --clip_sigma $CLIP_SIGMA \
  --blur_kernel $BLUR_KERNEL \
  --blur_sigma  $BLUR_SIGMA \
  --x_t_rate $XTRATE \
  --per_channel False
```

Given that the sampling process may take much time, we've included a way to split the sampling into multiple processes. To use this feature, include the flag `--num_chunks C`, where `C` is the number of chunks to split the dataset. Then, run `C` times the code using the flag `--chunk c`, where `c` is the chunk to generate the evaluation (hence, `c \in {0, 1, ..., C - 1}`).

The results will be stored `OUTPUT_PATH`. This folder has the following structure:

```
OUTPUT_PATH ----- Original ---- Correct
              |             |
              |             --- Incorrect
              |
              |
              |
              --- Results ---- EXPNAME ---- (I/C)C ---- (I/C)CF ---- CF
                                                                 |
                                                                 --- Info
                                                                 |
                                                                 --- Noise
                                                                 |
                                                                 --- SM
```

This structure useful to experiment since we can change only the `EXPNAME` to refer to another experiment without changing the original images. The folder `Original` contains the correctly classified (misclassified) images in `Correct` (`Incorrect`). We resume the structure of the counterfactuals explanations (`Results/EXPNAME`) as: `(I/C)C`: (In/correct) classification. `(I/C)CF`: (In/correct) counterfactual. `CF`: counterfactual images. `Info`: Useful information per instance. `Noise`: Noisy instance at timestep $\tau$ of the input data. `SM`: Difference between the input and its counterfactual. All files in all folders will have the same identifier.



## Citation

If you found useful our code, please cite our work.

```
@article{guo2026maskdime,
  title={MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations},
  author={Guo, Changlu and Christensen, Anders Nymark and Dahl, Anders Bjorholm and Hannemose, Morten Rieger},
  journal={arXiv preprint arXiv:2602.18792},
  year={2026}
}
``` 

## Code Base

We based our repository on [https://github.com/guillaumejs2403/DiME]).
