DATASET='PIE'
DATADIR='/home/david/Projects/MultimodalUE/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 0.1 \
    --epochs 500 \
    --lr 1e-1 \
    --end_lr 1e-3 \
    --a_eps 1e-7 \
    --num_inducing_points 200 \
    --seed $SEED 

