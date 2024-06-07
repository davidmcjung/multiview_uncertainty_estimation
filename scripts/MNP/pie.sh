DATASET='PIE'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 250 \
    --lr_scheduler piecewise \
    --lr_epochs 100,200 \
    --lr_list 1e-3,1e-4,1e-5 \
    --seed $SEED \
    --clipnorm 1e-4 \
    --n_context_points 300 \
    --l2_mlp 1e-3 \
    --mlp_norm_type layer_norm \
    --temp 0.1 \
    --warmup_epochs 100 \
    --warmup_lr 1e-2

