torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        distributed_train.py \
        --sync_bn \
        --n_epochs 100 \
        --batch_size 16 \
        --project new_cut \
        --use_wandb \
        --dataroot /data/seatgenfinaldestination5/ \
        --checkpoints_dir checkpoints/ \
        --name new_cut/ \
        "$@"
