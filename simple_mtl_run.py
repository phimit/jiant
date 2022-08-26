# TODO
#    x- simple -> normal run
#    /- external config
#    /- argparse for all arguments
#    - test/prediction mode

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.proj.simple.runscript as simple_run
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
from decode_preds import convert_prediction_to_disrpt
#from datasets import load_dataset_builder
import os
import argparse
from codecarbon import EmissionsTracker


EXP_DIR = "/home/muller/Devel/jiant/exp"
DATA_DIR = os.path.join(EXP_DIR,"tasks")

parser = argparse.ArgumentParser()

parser.add_argument("tasks",help="a string listing some tasks")
parser.add_argument("--run-name",default=None,help="name of the directory where to log experiments; if not set, will combine task and model")
parser.add_argument("--model-name",default="bert-base-multilingual-uncased",help="name of the model to use")
parser.add_argument("--model-path",default=None,help="path to the model; if model-name is on hugging face, this does not need to be set")
parser.add_argument("--exp-dir",default=EXP_DIR,help="directory where to find data and configs")

# todo: add batch size, epochs, eval_every_step, sth to set early stopping too
#       and an option for val/testing -> for test, needs to hack the task_config_path cos of error in metrics for test set
parser.add_argument("--batch-size",default=64,type=int,help="")
parser.add_argument("--epochs",default=1,type=float,help="nb of epochs for training")
parser.add_argument("--eval-every-step",default=100,type=int,help="")
parser.add_argument("--no_improvements_for_n_evals",default=5,type=int,
                    help="early stopping after n evals w/o improvements; needs eval-every step to be set")
parser.add_argument("--co2",action="store_true",default=False,help="track co2 emissions (needs internet access)")
#example model names: "bert-base-multilingual-uncased", "roberta-base"

args = parser.parse_args()
EXP_DIR = args.exp_dir
DATA_DIR = os.path.join(EXP_DIR,"tasks")
TASK_NAMES = args.tasks
if args.model_path is None:
    HF_PRETRAINED_MODEL_NAME = args.model_name 
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME
else: 
    HF_PRETRAINED_MODEL_NAME = args.model_name
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]

RUN_NAME = args.run_name if args.run_name is not None else f"run_{TASK_NAMES}_{MODEL_NAME}"

CO2_tracking = args.co2

if CO2_tracking: tracker = EmissionsTracker()

task_list = TASK_NAMES.split()

# testing the data reader
for task_name in task_list:
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=os.path.join(DATA_DIR,f"configs/{task_name}_config.json"),
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        output_dir=f"./cache/{task_name}",
        phases=["train", "val"],
        smart_truncate = True,
    ))

task_dev_caches = {task_name:caching.ChunkedFilesDataCache(f"./cache/{task_name}/val") for task_name in task_list}

#chunk = caching.ChunkedFilesDataCache("./cache/disrpt21_gum/train").load_chunk(0)#[0]["data_row"]
#for one in chunk[:3]:
#    #print(one["metadata"])
#    row = one["data_row"]




SIMPLE = False


if CO2_tracking: tracker.start()
if SIMPLE: 
    args = simple_run.RunConfiguration(
            run_name=RUN_NAME,
            exp_dir=EXP_DIR,
            data_dir=DATA_DIR,
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            #task_config_base_path=os.path.join(DATA_DIR,"/tasks/configs"),
            #tasks=TASK_NAME.split(),
            train_task_name_list=TASK_NAMES.split(),
            val_task_name_list=TASK_NAMES.split(),
            eval_every_steps=args.eval_every_step,
            train_batch_size=args.batch_size,
            num_train_epochs=30,
            write_val_preds=True,
            
        )
    simple_run.run_simple(args)
else:
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=os.path.join(DATA_DIR,f"configs/"),
        task_cache_base_path="./cache",
        train_task_name_list=TASK_NAMES.split(),
        val_task_name_list=TASK_NAMES.split(),
        train_batch_size=args.batch_size, # tony = 2!
        eval_batch_size=8,
        epochs=args.epochs,
        num_gpus=1,
    ).create_config()
    os.makedirs(os.path.join(EXP_DIR,"run_configs/"), exist_ok=True)
    py_io.write_json(jiant_run_config,os.path.join(EXP_DIR,"run_configs/jiant_run_config.json"))
    display.show_json(jiant_run_config)

    print("looking for model at : ",os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"))

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=os.path.join(EXP_DIR,"run_configs/jiant_run_config.json"),
        output_dir=os.path.join("runs",RUN_NAME),
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        model_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"),
        model_config_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"/config.json"),
        learning_rate=1e-5, # tony = 1e-3
        eval_every_steps=args.eval_every_step,
        no_improvements_for_n_evals=args.no_improvements_for_n_evals,
        write_val_preds=True,
        do_train=True,
        do_val=True,
        do_save=True,
        force_overwrite=True,
    )

    main_runscript.run_loop(run_args)

    # predictions are stored only as torch tensors, this puts them in disrpt format
    # right now this only provably works for one task at a time
    # TODO: check pred file in multi-task exps
    infile = os.path.join("runs",RUN_NAME,"val_preds.p")
    for one_task in task_list:
        outfile = os.path.join("runs",RUN_NAME,one_task+"_dev.disrpt")
        convert_prediction_to_disrpt(infile,one_task,outfile,task_dev_caches[one_task])


if CO2_tracking: tracker.stop()
