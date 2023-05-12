Instructions for running Melodi's Discret system for task 1 and 2 --- Disrpt 2023 
=====================================================================================

## Installation 

-1 clone the repo somewhere

```
git clone git@github.com:phimit/jiant.git
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
Change the first two lines of setup_disrpt23.sh to point to your local installation + where data are located
then 

```
bash scripts/setup_disrpt23.sh
```

## Training TASK 1 + 2 + evaluation
```
bash scripts/run_disrpt23.sh
```