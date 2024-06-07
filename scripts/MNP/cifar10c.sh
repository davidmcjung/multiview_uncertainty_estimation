DATASET='CIFAR10-C'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 15 \
    --lr_scheduler piecewise \
    --lr_epochs 10 \
    --lr_list 5e-4,1e-4 \
    --seed $SEED \
    --clipnorm 1e-1 \
    --n_context_points 200 \
    --l2_mlp 1e-2 \
    --l2_lengthscale 1e-8 \
    --mlp_norm_type batch_norm \
    --temp 0.01 \
    --warmup_epochs 30 \
    --r_dim 1024 \
    --warmup_lr 1e-3 \
    --context_memory_init random 


