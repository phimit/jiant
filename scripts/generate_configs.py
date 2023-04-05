"""generate config files for jiant multi-task setup 

given a dir with datasets (eg disrpt23/data_clean)
and a destination (eg exp/tasks/configs/disrpt23)
generate a config file for each separate dataset: one for .conllu, one for .split

Example call to script: 
python generate_configs.py /home/muller/Devel/jiant/exp/tasks/data/disrpt23/ /home/muller/Devel/jiant/exp/tasks/configs/disrpt23/

"""
from string import Template

# example
example = """
{
  "task": "disrpt21",
  "paths": {
    "train": "/home/muller/Devel/jiant/exp/tasks/data/disrpt23/fra.sdrt.annodis/fra.sdrt.annodis_train.conllu",
    "val": "/home/muller/Devel/jiant/exp/tasks/data/disrpt23/fra.sdrt.annodis/fra.sdrt.annodis_dev.conllu",
    "test": "/home/muller/Devel/jiant/exp/tasks/data/disrpt23/fra.sdrt.annodis/fra.sdrt.annodis_test.conllu"
  },
  "name": "disrpt23_fra.sdrt.annodis",
  "kwargs": {
    "corpus":"fra.sdrt.annodis",
    "recurrent_layer": "None",
    "rnn_config": {
        "hidden_size":100,
        "bidirectional":true,
        "batch_first":true,
        "bias":false,
	"dropout":0.5
    }
  }
}
"""

# warning: disrpt21/disrpt21_conn are the generic name of disrpt tasks defined within jiant
# this could be renamed but frankly not urgent
# rnn config won't be used, but is here as example of a configuration that can be passed on to the task initialization in jiant
# (see relevant task head code in jiant/jiant/proj/main/modeling/heads.py)
cfg_template = Template("""
{
  "task": "${task_name}",
  "paths": {
    "train": "${data_dir}/${corpus_name}/${corpus_name}_train.${input_type}",
    "val": "${data_dir}/${corpus_name}/${corpus_name}_dev.${input_type}",
    "test": "${data_dir}/${corpus_name}/${corpus_name}_test.${input_type}"
  },
  "name": "disrpt23_${corpus_task}",
  "kwargs": {
    "corpus":"${corpus_name}",
    "recurrent_layer": "None",
    "rnn_config": {
        "hidden_size":100,
        "bidirectional":true,
        "batch_first":true,
        "bias":false,
	"dropout":0.5
    }
  }
}
""")

l_input_types = {"conllu","split"}

test_example = {
    "task_name":"disrpt21",
    "data_dir":"/home/muller/Devel/jiant/exp/tasks/data/disrpt23",
    "corpus_name":"fra.sdrt.annodis",
    "input_type":"conllu"
    }

#assert example==cfg_template.substitute(test_example)


if __name__=="__main__":
    from glob import glob
    import os, os.path
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",help="dir name where all datasets resides")
    parser.add_argument("output_dir",help="dir where all configs will be written")

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    params = {"data_dir":data_dir}


    subcorpora_paths = [x for x in glob(os.path.join(data_dir,'*')) if os.path.isdir(x)]
    print(subcorpora_paths)
    for subcorpus_path in subcorpora_paths:
        #print("looking at : ",subcorpus_path)
        # check that this is a subcorpus, whose name must be lang.framework.name
        subcorpus_name = os.path.basename(subcorpus_path)
        if not(len(subcorpus_name.split("."))==3):
            print("problem ?",subcorpus_name)
            continue
        lang, framework, name = subcorpus_name.split(".")
        if framework == "pdtb":
            params["task_name"] = "disrpt21_conn"
        else: 
            params["task_name"] = "disrpt21"

        params["corpus_name"] = subcorpus_name
        subtaskname = subcorpus_name.replace(".","_")
        params["corpus_task"] = subtaskname
        
        for setup in l_input_types: 
            params["input_type"] = setup    
            config = cfg_template.substitute(params)
            # _config obligatory because of jiant default naming when saving/caching config
            outfile_name = os.path.join(output_dir,f"disrpt23_{subtaskname}_{setup}_config.json")
            print("writing ",outfile_name)
            outfile = open(outfile_name,"w")
            print(config,file=outfile)

