export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

export MODEL=xlm-roberta-large
#export MODEL=bert-base-multilingual-uncased
#export MODEL=bigscience/bloom-1b1


# set parameters for XLMlarge vs XLMbase or BERT
[[ $MODEL == "xlm-roberta-large" ]] \
    && BATCH_SIZE=1 \
	|| BATCH_SIZE=16

[[ $MODEL == "xlm-roberta-large" ]] \
    && FROZEN="0-5" \
	|| FROZEN="0-2"

export EPOCHS=0.1



for idx in {1..2}; 
#for dataset in disrpt23_spa_rst_merged_conllu disrpt23_eng_rst_merged_conllu disrpt23_zho_rst_merged_conllu; 
do
 echo "training on ..."  disrpt23_fra_sdrt_cid${idx}_split
 python simple_mtl_run.py disrpt23_fra_sdrt_cid${idx}_split --model-name $MODEL --epochs $EPOCHS --eval-every-step 500 \
   --batch-size $BATCH_SIZE --gradient-accumulation-steps 4 \
	 --sampling-strategy ProportionalMultiTaskSampler --max-seq-length 200 \
	 --config-dir exp/tasks/configs/disrpt23 --freeze-layers $FROZEN --test
done; 


# for dataset in ${OOD_CONLLU[@]}; 
# do
# echo "training on ..." $dataset
# set -- $dataset
# set -x
# echo "=== and doing OOD task" $2 
# python simple_mtl_run.py "$1" --model-name $MODEL --epochs $EPOCHS --eval-every-step 500 \
# 	--batch-size $BATCH_SIZE --gradient-accumulation-steps 4 \
# 	--no_improvements_for_n_evals 10 --sampling-strategy ProportionalMultiTaskSampler --max-seq-length 180\
# 	--config-dir exp/tasks/configs/disrpt23 --freeze-layers $FROZEN --ood $2 --test        
# done; 
