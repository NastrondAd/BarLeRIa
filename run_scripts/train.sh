num_gpus=$1
master_port=$2
dataset_name=$3 # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name=$4
output_dir=$5
python -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_port ${master_port} train.py \
      --config config/$dataset_name/$config_name \
      --opts TRAIN.output_folder $output_dir \

