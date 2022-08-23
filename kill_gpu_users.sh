nvidia-smi | grep python3 | awk -v gpu=$1 '{if($2==gpu) {print $5}}' | xargs kill
