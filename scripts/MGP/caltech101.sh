DATASET='Caltech101'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 1.0 \
    --epochs 600 \
    --lr 1e-2 \
    --end_lr 1e-2 \
    --a_eps 1e-7 \
    --num_inducing_points 200 \
    --seed $SEED \
    --noinit_lengthscale_data


