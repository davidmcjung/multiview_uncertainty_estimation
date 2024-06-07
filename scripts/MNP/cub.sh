DATASET='CUB'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 50 \
    --lr 1e-3 \
    --end_lr 1e-3 \
    --seed $SEED \
    --clipnorm 1e-1 \
    --n_context_points 200 \
    --l2_mlp 3e-2 \
    --mlp_norm_type layer_norm \
    --temp 0.01 \
    --warmup_epochs 100 \
    --warmup_lr 1e-2 \
    --update_freq 50


