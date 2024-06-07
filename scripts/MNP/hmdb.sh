DATASET='HMDB'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 30 \
    --lr_scheduler piecewise \
    --lr_epochs 10 \
    --lr_list 5e-4,1e-5 \
    --seed $SEED \
    --n_context_points 800 \
    --l2_mlp 1e-7 \
    --l2_lengthscale 1e-7 \
    --mlp_norm_type batch_norm \
    --temp 0.01 \
    --warmup_epochs 100 \
    --warmup_lr 1e-3 \
    --r_dim 1024 \
    --feat_norm_type standard \
    --init_lengthscale 20 \
    --norand_mask \
    --context_memory_init random \
    --n_z_samples 10


