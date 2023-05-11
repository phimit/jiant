########## (D) training and prediction ####################

###### D.1 corpus grouping #########
# skipped

###### D.2 training+pred (+ood) ########
bash train_conllu.sh
bash train_split.sh
bash train_ood.sh
#bash train_merged.sh
###### D3. move predictions at same place and evaluate
mkdir $EXP_DIR/predictions
mkdir $EXP_DIR/predictions/conllu
mkdir $EXP_DIR/predictions/split
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/conllu --task-type conllu --model roberta-large --target dev
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/conllu  --task-type conllu --model roberta-large --target test
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/split --task-type tok --model roberta-large --target dev
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/split --task-type tok --model roberta-large --target test

### call seg_eval and put everything in a file
### if this script is run multiple times, this will generate a different filename
