#disrpt21_annodis_config.json  disrpt21_cstn_config.json  disrpt21_pcc_config.json   disrpt21_rstdt_config.json   disrpt21_sctb_config.json
#disrpt21_cdtb_config.json     disrpt21_gum_config.json   disrpt21_pdtb_config.json  disrpt21_rststb_config.json  disrpt21_tdb_config.json

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export MODEL=xlm-roberta-base
#export MODEL=bert-base-multilingual-cased
#export MODEL=bigscience/bloom-1b1

for dataset in disrpt21_pdtb disrpt21_tdb disrpt21_cdtb ; # disrpt21_cstn';
do
    # does not work for multitask setup
    #rm -rf ./cache/$dataset
    echo "training on ..." $dataset
    python simple_mtl_run.py "${dataset}" --co2 --model-name $MODEL --epochs 30 --eval-every-step 500 \
                                --batch-size 16 --gradient-accumulation-steps 4 --no_improvements_for_n_evals 10 \
                                --sampling-strategy ProportionalMultiTaskSampler --max-seq-length 180
    # move final dev evaluation to the sub-run dir
done; 
