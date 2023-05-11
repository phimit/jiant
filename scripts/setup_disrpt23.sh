######### (B) configuration #################
# only these lines should have to be changed
export JIANT_DIR=/home/muller/Devel/jiant/
ln -s /path/to/disrpt_data_dir ${JIANT_DIR}/exp/tasks/data/disrpt23

####################################
export PYTHONPATH=$JIANT_DIR:$PYTHONPATH
# locating disrpt23 datasets 
export DISRPT23_DATADIR=${JIANT_DIR}/exp/tasks/data/disrpt23
# setting dir for experiment result
export EXP_DIR=${JIANT_DIR}/exp/
mkdir $EXP_DIR/tasks/configs/disrpt23
# generating configuration scripts
python $JIANT_DIR/scripts/generate_configs.py $DISRPT23_DATADIR $EXP_DIR/tasks/configs/disrpt23
# loading the necessary models
mkdir $EXP_DIR/models
python $JIANT_DIR/scripts/install_models.py xlm-roberta-base
python $JIANT_DIR/scripts/install_models.py xlm-roberta-large
# creating space to save results
mkdir $EXP_DIR/runs