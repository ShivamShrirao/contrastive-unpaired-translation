pip3 install -r code/requirements.txt

git config --global --add safe.directory '*'
git config --global --add safe.directory /opt/ml/code

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        distributed_train.py \
        "$@"
