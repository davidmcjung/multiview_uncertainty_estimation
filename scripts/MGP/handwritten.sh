DATASET='Handwritten'
DATADIR='/home/david/Projects/MultimodalUE/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 0.5 \
    --epochs 500 \
    --lr 1e-2 \
    --end_lr 1e-4 \
    --a_eps 1e-7 \
    --num_inducing_points 300 \
    --seed $SEED 


