python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        distributed_train.py \
        --sync_bn \
        --n_epochs 100 \
        --batch_size 8 \
        --project new_cut \
        --use_wandb \
        --dataroot ~/data/nerf_refine_data \
        --checkpoints_dir checkpoints/new_cut/ \
        "$@"
