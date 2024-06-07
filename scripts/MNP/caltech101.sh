DATASET='Caltech101'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 70 \
    --lr_scheduler piecewise \
    --lr_epochs 10,30,80 \
    --lr_list 1e-3,5e-4,1e-4,1e-6 \
    --seed $SEED \
    --clipnorm 1e-1 \
    --n_context_points 700 \
    --l2_mlp 1e-3 \
    --mlp_norm_type batch_norm \
    --temp 0.01 \
    --warmup_epochs 100 \
    --warmup_lr 1e-3 \
    --r_dim 1024 \
    --feat_norm_type standard


