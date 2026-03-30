source ~/.bashrc
source activate maskdime

module load gcc/10.5.0-binutils-2.40
module load mpi/4.0.7-gcc-10.5.0-binutils-2.40

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
OUTPUT_PATH="maskdime"
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



python -W ignore main.py \
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





