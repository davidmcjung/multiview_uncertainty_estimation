DATASET='CIFAR10-C'
DATADIR='/home/david/Projects/MultimodalUE/data'
SEED=1

python main.py \
    --dataset $DATASET \
    --data_dir $DATADIR \
    --model MGP \
    --beta 2.0 \
    --epochs 100 \
    --lr 1e-2 \
    --end_lr 1e-5 \
    --a_eps 1e-3 \
    --num_inducing_points 200 \
    --seed $SEED 

