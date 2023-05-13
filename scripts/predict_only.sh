### call trained jiant model for inference on validation or test set of a specific task 
### these must have been defined during task definition or via a specific out-of-domain configuration
### not usable to just call the model (will be in another script ... in progress)
### normally this should not be called "manually" unless to fix something that went wrong

### Inspired from https://github.com/phimit/jiant/blob/master/jiant/scripts/benchmarks/xtreme/subscripts/e_run_models.sh
### with modification found in https://github.com/phimit/jiant/blob/master/guides/projects/xstilts.md
# NB: documentation above is not up to date with actual repo.

# FIXME: 
#    x- run_config should come from actual experiment to preserve all parameters; fixed here for testing
#    x- fix proper paths for everythin
# TODO:
#    - pass everything as arguments of the script: base_path / model_type / cuda_device

# EXAMPLE USAGE
# >> bash scripts/predict_only.sh /home/muller/Devel/jiant//runs/run_disrpt23_eng_rst_gum_conllu_xlm-roberta-large/1683703080/ disrpt23_eng_rst_gum_conllu
# will run the best model found in that experiment run on the validation+test set, producing val_preds.p + test_preds.p
# 

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


export BASE_PATH=$JIANT_DIR
export MODEL_TYPE=xlm-roberta-large
export TASK=$2   # eg disrtp23_deu_rst_pcc_split 
export MODEL_PATH=$1
export PYTHONPATH=$BASE_PATH:$PYTHONPATH

set -x
# fix the run_config file which prolly doesnt include the test file reference if we have to do this after training
python $JIANT_DIR/scripts/patch_test_config.py ${MODEL_PATH} $TASK
# launch predictions
python $JIANT_DIR/jiant/proj/main/runscript.py \
        run_with_continue \
        --ZZsrc ${BASE_PATH}/exp/models/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${MODEL_PATH}/run_config.json \
        --ZZoverrides model_path \
        --model_load_mode partial \
        --model_path ${MODEL_PATH}/best_model.p \
        --force_overwrite \
        --do_val \
        --output_dir ${MODEL_PATH}\
        --write_val_preds \
        --write_test_preds

# generate disrpt files from the prediction matrices
# 
python $JIANT_DIR/decode_preds.py --test ${MODEL_PATH}/test_preds.p $TASK
mv ${TASK}_test.txt ${MODEL_PATH}
# if we have to do this by hand, the test prediction should be here
# (shouldnt happen) but if they were generated during training they are in ${MODEL_PATH}/../test_preds.p
#python $JIANT_DIR/decode_preds.py --test ${MODEL_PATH}/test_preds.p ${MODEL_PATH}/../$2 
