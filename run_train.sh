# training script
model=vit_base_patch14_reg4_dinov2.lvd142m
root=./snapshot
dataroot=/data/Datasets

torchrun \
--nnodes=1 \
--nproc_per_node=4 \
./train_net.py \
--model_arch ${model} \
--pretrained_start_weights \
--data_path ${dataroot}/CUB_200_2011 \
--batch_size 4 \
--epochs 25 \
--dataset cub \
--save_every_n_epochs 8 \
--num_workers 2 \
--image_sub_path_train images \
--image_sub_path_test images \
--train_split 1 \
--eval_mode test \
--snapshot_dir ${root} \
--lr 1e-6 \
--optimizer_type adam \
--scheduler_type steplr \
--scheduler_gamma 0.5 \
--scheduler_step_size 4 \
--scratch_lr_factor 1e4 \
--modulation_lr_factor 1e4 \
--finer_lr_factor 2e2 \
--drop_path 0.0 \
--smoothing 0 \
--augmentations_to_use cub_original \
--image_size 518 \
--num_parts 4 \
--weight_decay 0 \
--classification_loss 1 \
--presence_loss 1 \
--equivariance_loss 1 \
--total_variation_loss 1 \
--enforced_presence_loss 1 \
--enforced_presence_loss_type enforced_presence \
--pixel_wise_entropy_loss 1 \
--gumbel_softmax \
--freeze_backbone \
--presence_loss_type original \
--modulation_type layer_norm \
--modulation_orth \
--grad_norm_clip 2.0 \
--n_pro 17,14,11,8,5