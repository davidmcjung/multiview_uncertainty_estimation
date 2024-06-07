DATASET='Handwritten'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MNP \
    --epochs 5 \
    --lr_scheduler piecewise \
    --lr_epochs 15,50 \
    --lr_list 3e-4,3e-5,3e-6 \
    --seed $SEED \
    --clipnorm 1e-3 \
    --n_context_points 100 \
    --l2_mlp 2e-5 \
    --mlp_norm_type layer_norm \
    --temp 0.25 \
    --warmup_epochs 100 \
    --warmup_lr 1e-2 \


