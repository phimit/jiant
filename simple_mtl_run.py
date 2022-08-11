# TODO
#    - simple -> normal run
#    - external config
#    - argparse for all arguments

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.proj.simple.runscript as simple_run
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
#from datasets import load_dataset_builder
import os
import argparse

TASK_NAME = "disrpt21_gum"
HF_PRETRAINED_MODEL_NAME = "bert-base-multilingual-uncased"#"roberta-base"
MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]
RUN_NAME = f"simple_{TASK_NAME}_{MODEL_NAME}"
EXP_DIR = "/home/muller/Install/jiant/exp"
DATA_DIR = os.path.join(EXP_DIR,"tasks")

# testing the data reader
for task_name in [TASK_NAME]:
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=os.path.join(DATA_DIR,f"configs/{task_name}_config.json"),
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        output_dir=f"./cache/{task_name}",
        phases=["train", "val"],
    ))


#chunk = caching.ChunkedFilesDataCache("./cache/disrpt21_gum/train").load_chunk(0)#[0]["data_row"]
#for one in chunk[:3]:
#    #print(one["metadata"])
#    row = one["data_row"]

SIMPLE = True

if SIMPLE: 
    args = simple_run.RunConfiguration(
            run_name=RUN_NAME,
            exp_dir=EXP_DIR,
            data_dir=DATA_DIR,
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            tasks=TASK_NAME,
            train_batch_size=64,
            num_train_epochs=3,
            write_val_preds=True,
            
        )
    simple_run.run_simple(args)
else:
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=os.path.join(DATA_DIR,f"configs/"),
        task_cache_base_path="./cache",
        train_task_name_list=["disrpt21_gum"],
        val_task_name_list=["disrpt21_gum"],
        train_batch_size=4,
        eval_batch_size=8,
        epochs=0.5,
        num_gpus=1,
    ).create_config()
    os.makedirs(os.path.join(EXP_DIR,"run_configs/"), exist_ok=True)
    py_io.write_json(jiant_run_config, "./run_configs/jiant_run_config.json")
    display.show_json(jiant_run_config)