# interpretability evaluation
model=vit_base_patch14_reg4_dinov2.lvd142m
root=./snapshot
dataroot=/root/datasets

python evaluate_consistency.py \
--model_path ${root}/snapshot_best.pt \
--dataset cub \
--center_crop \
--eval_mode nmi_ari \
--num_parts 4 \
--model_arch ${model} \
--data_path ${dataroot}/cub200_cropped \
--batch_size 8 \
--num_workers 2 \
--image_size 518 \
--modulation_type layer_norm \
--half_size 36 \
--n_pro 17,14,11,8,5 \
--output_dir ${root}/output