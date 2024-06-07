DATASET='Scene15'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 1.0 \
    --epochs 400 \
    --lr 1e-1 \
    --end_lr 5e-5 \
    --a_eps 1e-5 \
    --num_inducing_points 300 \
    --seed $SEED \
    --noinit_lengthscale_data


