DATASET='CUB'
DATADIR='/path/to/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 1.0 \
    --epochs 200 \
    --lr 1e-1 \
    --end_lr 1e-5 \
    --a_eps 1e-7 \
    --num_inducing_points 100 \
    --seed $SEED \
    --test_weighting softmax_entropy


