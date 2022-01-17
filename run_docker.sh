NETWORK=$1
docker build -t tf_adequatedl . &&
  docker run \
    --rm --gpus all -it \
    -v $PWD:/workspace \
    -v /home/edupuis@inl34.ec-lyon.fr/.onnx/datasets:/datasets \
    -w /workspace \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    tf_adequatedl \
    python3 -u main.py \
    --network_dataset ${NETWORK} \
    --log_file "logs/${NETWORK}/latest" \
    --num_samples 5513 \
    --dataset_scale 0.1 \
    --with_local_optimization 1 \
    --mode exploration \
    --optimization two_step_optimization

