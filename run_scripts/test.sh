num_gpus=$1
master_port=$2
dataset_name=$3 # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name=$4
output_dir=$5
for split_name in val testA testB test
do
# Evaluation on the specified of the specified dataset
      CUDA_VISIBLE_DEVICES=$gpu python -u test.py \
            --config config/$dataset_name/$config_name \
            --opts TEST.test_split $split_name \
                  TEST.test_lmdb datasets/lmdb/$dataset_name/$split_name.lmdb
done