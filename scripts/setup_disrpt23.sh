######### (B) configuration #################
# (B) 
export JIANT_DIR=/home/muller/Devel/jiant/
# locating disrpt23 datasets 
export DISRPT23_DATADIR=${JIANT_DIR}/exp/tasks/data/disrpt23
# setting dir for experiment result
export EXP_DIR=${JIANT_DIR}/exp/
mkdir $EXP_DIR/tasks/configs/disrpt23
# generating configuration scripts
python $JIANT_DIR/scripts/generate_configs.py $DISRPT23_DATADIR $EXP_DIR/tasks/configs/disrpt23
# loading the necessary models
mkdir $EXP_DIR/models
python $JIANT_DIR/install_models.py xlm-roberta-base
# creating space to save results
mkdir $EXP_DIR/runs