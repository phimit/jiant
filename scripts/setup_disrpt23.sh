######### (B) configuration #################
# only these lines should have to be changed
export JIANT_DIR=/home/muller/Devel/jiant
# the data directory is assumed to be already preprocessed to replace "_" for corpora with licences; 
# it assumes also all all standard+suprise datasets are in the same directory
ln -s /moredata/disrpt/data_clean_all/ ${JIANT_DIR}/exp/tasks/data/disrpt23

####################################
export PYTHONPATH=$JIANT_DIR:$PYTHONPATH
# locating disrpt23 datasets 
export DISRPT23_DATADIR=${JIANT_DIR}/exp/tasks/data/disrpt23
# setting dir for experiment result
export EXP_DIR=${JIANT_DIR}/exp/
mkdir -p $EXP_DIR/tasks/configs/disrpt23
# generating configuration scripts for the infrastructure
python $JIANT_DIR/scripts/generate_configs.py $DISRPT23_DATADIR $EXP_DIR/tasks/configs/disrpt23
# loading the necessary models; if the setup is interrupted after this for any reason, you have to delete those
# before relaunching setup
mkdir -p $EXP_DIR/models
#python $JIANT_DIR/scripts/install_model.py xlm-roberta-base
python $JIANT_DIR/scripts/install_model.py xlm-roberta-large
# creating space to save results
mkdir -p $JIANT_DIR/runs
# preprocessing .tok files to apply a sentence splitter and generate .split files 
python $JIANT_DIR/scripts/split_sentences.py -m trankit --gpu $DISRPT23_DATADIR
