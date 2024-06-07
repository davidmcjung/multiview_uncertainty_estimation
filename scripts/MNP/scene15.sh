DATASET='Scene15'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 200 \
    --lr_scheduler piecewise \
    --lr_epochs 40 \
    --lr_list 5e-4,1e-4 \
    --seed $SEED \
    --clipnorm 1e-4 \
    --n_context_points 300 \
    --l2_mlp 1e-4 \
    --l2_lengthscale 1e-4 \
    --mlp_norm_type batch_norm \
    --temp 0.5 \
    --r_dim 512 \
    --warmup_epochs 100 \
    --warmup_lr 1e-2 \
    --feat_norm_type standard


