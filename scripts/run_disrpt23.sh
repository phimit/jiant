########## (D) training and prediction ####################
# This needs the following env variables, defined in setup
# if you launch the training separately from setup
# you have to reset them here, eg
#export JIANT_DIR=/home/muller/Devel/jiant
#export EXP_DIR=${JIANT_DIR}/exp/
###### D.1 corpus grouping #########
# skipped

###### D.2 training+pred (+ood) ########
bash train_conllu.sh
bash train_split.sh
bash train_ood.sh
#bash train_merged.sh
###### D3. move predictions at same place and evaluate
#mkdir -p $EXP_DIR/predictions
mkdir -p $EXP_DIR/predictions/conllu
mkdir -p $EXP_DIR/predictions/split
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/conllu --task-type conllu --model roberta-large --target dev
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/conllu  --task-type conllu --model roberta-large --target test
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/split --task-type tok --model roberta-large --target dev
python $JIANT_DIR/scripts/collect_preds.py $JIANT_DIR/runs/ $EXP_DIR/predictions/split --task-type tok --model roberta-large --target test

### call seg_eval and put everything in a file
### if this script is run multiple times, this will generate a different filename
mkdir -p scores
python $JIANT_DIR/scripts/task_1_2_scores.py  ${JIANT_DIR}/exp/tasks/data/disrpt23/  $EXP_DIR/predictions/conllu/ --outdir scores/ --task-type conllu
python $JIANT_DIR/scripts/task_1_2_scores.py  ${JIANT_DIR}/exp/tasks/data/disrpt23/  $EXP_DIR/predictions/split/ --outdir scores/ --task-type tok
