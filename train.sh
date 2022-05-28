torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        distributed_train.py \
        --n_epochs 150 \
        --batch_size 2 \
        --project body_cut \
        --use_wandb True \
        --name no_windows/ \
        "$@"
