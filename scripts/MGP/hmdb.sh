DATASET='HMDB'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 1.0 \
    --epochs 500 \
    --lr 1e-1 \
    --end_lr 1e-3 \
    --a_eps 1e-5 \
    --num_inducing_points 200 \
    --seed $SEED \


