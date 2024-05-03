export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


export BASE_MODEL=xlm-roberta-large
export TRAINED_MODEL=$2
export dataset=$1

set -x
python simple_mtl_run.py "${dataset}" --predict-only --model-name $BASE_MODEL --model-path $TRAINED_MODEL --max-seq-length 180 \
	 --config-dir exp/tasks/configs/disrpt23 
 
