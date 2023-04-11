################################################
# General script/instructions to run disrpt23 experiments
# usint jiant for segmentation/connective detection
################################################

######### (A) Install ########
# (A) should be done out of this script since it needs to be done once
# A.1 clone the repo somewhere
"git clone git@github.com:phimit/jiant.git"
# A.2 create environment + install dependencies
"conda create -n discut23 python==3.10"
"cd jiant"
# ignore jiant requirements, which are not up to date
# on most systems, if you get the following running it should be enough
"pip install transformers pandas torch"


######### (B) configuration #################
# (B) should be done out of this script since it needs to be done once
export JIANT_DIR=/home/muller/jiant/
# locating disrpt23 datasets
export DISRPT23_DATADIR=/home/muller/jiant/exp/tasks/data/disrpt23
# setting dir for experiment result
export EXP_DIR=/home/muller/jiant/exp/
mkdir $EXP_DIR/tasks/configs/disrpt23
# generating configuration scripts
python $JIANT_DIR/generate_configs.py $DISRPT23_DATADIR $EXP_DIR/tasks/configs/disrpt23
# loading the necessary models
mkdir $EXP_DIR/models
python $JIANT_DIR/install_models.py xlm-roberta-base
# creating space to save results
mkdir $EXP_DIR/runs

######### (C) Preprocessing ###########################
# (C) should be done only once for the final experiment
# apply a sentence splitter to every .tok -> produces a .split file (== silver corpus)
# (path to script needs to be updated for final version)
python split_sentences.py -m trankit --gpu $DISRPT23_DATADIR
