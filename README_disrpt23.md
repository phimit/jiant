Instructions for running Melodi's Discut system for task 1 and 2 --- Disrpt 2023 
=====================================================================================

## Installation 

-1 clone the repo somewhere

```
git clone git@github.com:phimit/jiant.git
```
or
```
git clone https://github.com/phimit/jiant.git
```

-2 create environment + install dependencies

```
conda create -n discut23_test python==3.10
cd jiant
conda activate disrpt23_test
```

 ignore jiant requirements, which are not up to date / might crash
 on most systems, if you get the following running it should be enough

```
pip install transformers pandas torch attrs seqeval Levenshtein nltk trankit numexpr codecarbon
```

## Setup 
You have to modif the script before running it: 
Change the first two lines of setup_disrpt23.sh to point to your local installation + where data are located
then 

```
bash scripts/setup_disrpt23.sh
```

## Training TASK 1 + 2 + evaluation
NB: This uses environment variables defined in setup; if you make a run in a different terminal, you have to reset them in run_disrpt23.sh (cf comments in the script)

```
bash scripts/run_disrpt23.sh
```
If you have access to multiple GPUs and want to restrict to a given card
you can type e.g. 

```
CUDA_VISIBLE_DEVICES=0 bash scripts/run_disrpt23.sh
```

Once this is done, you'll have

  - runs in runs/ with logs, models
  - predictions in exp/predictions/
  - all scores listed in scores/ directly

