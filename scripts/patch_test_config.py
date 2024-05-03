"""
add entries to a run_configs to have test predictions when called on a val-only run (see predict-only)

usage:

patch_test_config run_path task_name

eg
"""
import sys
import os
import json


example_target = {
"task_cache_config_dict": {"test": "./cache/disrpt23_eng_rst_gum_split/test"},
"task_run_config": {"test_task_list": ["disrpt23_eng_rst_gum_split"]
  },
}

run_path = sys.argv[1]
task_name = sys.argv[2]

config_file = os.path.join(run_path,"run_config.json")
fp = open(config_file)
cfg_dict = json.load(fp)

cfg_dict["task_cache_config_dict"][task_name]["test"] = f"./cache/{task_name}/test"
cfg_dict["task_run_config"]["test_task_list"] = [task_name]

fp.close()
fp = open(config_file,"w")
json.dump(cfg_dict,fp,indent=4, separators=(',', ': '), sort_keys=False)