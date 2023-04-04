# TODO
#    x- simple -> normal run
#     - config disrtp23 : conllu / split
#        FIXME : bug in saving config file for expe (name hardcoded in jiant/proj/simple/runscript.py (function create_and_write_task_configs)
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
DATA_DIR = os.path.join(EXP_DIR,"tasks","configs")

parser = argparse.ArgumentParser()

parser.add_argument("tasks",help="a string listing some tasks")
parser.add_argument("--run-name",default=None,help="name of the directory where to log experiments; if not set, will combine task and model")
parser.add_argument("--model-name",default="bert-base-multilingual-uncased",help="name of the model to use")
parser.add_argument("--model-path",default=None,help="path to the model; if model-name is on hugging face, this does not need to be set")
parser.add_argument("--exp-dir",default=EXP_DIR,help="directory where to find data and configs")
parser.add_argument("--config-dir",default=DATA_DIR,help="directory where to find task configs, default is EXP_DIR/tasks/configs")
#parser.add_argument("--input-type",choices=["conllu","split"],default="conllu",help="run on gold sentences (conllu) or automatically sentence-split (split)")

# todo: add batch size, epochs, eval_every_step, sth to set early stopping too
#       and an option for val/testing -> for test, needs to hack the task_config_path cos of error in metrics for test set
parser.add_argument("--max-seq-length",default=128,type=int,help="max nb of (?sub)tokens before truncation of inputs")
parser.add_argument("--sampling-strategy",default="ProportionalMultiTaskSampler",help="task sampling strategy for multi-task learning; default: proportional")
# needs template so deactivated for now
#parser.add_argument("--use-config",default=None,help="use external config file for this experiment; overrides everything except what is not set")

# todo: 
#       an option for val/testing -> for test, needs to hack the task_config_path cos of error in metrics for test set
parser.add_argument("--batch-size",default=16,type=int,help="")
parser.add_argument("--gradient-accumulation-steps",default=4,type=int,help="delaying gradient update to allow for larger effective batches")
parser.add_argument("--epochs",default=1,type=float,help="nb of epochs for training")
parser.add_argument("--eval-every-step",default=100,type=int,help="")
parser.add_argument("--no_improvements_for_n_evals",default=5,type=int,
                    help="early stopping after n evals w/o improvements; needs eval-every step to be set")

parser.add_argument("--co2",action="store_true",default=False,help="track co2 emissions (needs internet access)")
parser.add_argument("--fp16",action="store_true",default=False,help="activate mixed precision 16/32bit; needs apex installed")

# dynamic learning rate: warmup_steps_proportion, eg 

#example model names: "bert-base-multilingual-uncased", "roberta-base"

args = parser.parse_args()
EXP_DIR = args.exp_dir
#DATA_DIR = os.path.join(EXP_DIR,"tasks")
DATA_DIR = args.config_dir
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
    #task_config_path=os.path.join(DATA_DIR,f"{task_name}_config.json")
    #print("config ?",task_config_path)
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=os.path.join(DATA_DIR,f"{task_name}_config.json"),
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        output_dir=f"./cache/{task_name}",
        max_seq_length=args.max_seq_length,
        phases=["train", "val"],
        smart_truncate = True,
    ))

task_dev_caches = {task_name:caching.ChunkedFilesDataCache(f"./cache/{task_name}/val") for task_name in task_list}

#chunk = caching.ChunkedFilesDataCache("./cache/disrpt21_gum/train").load_chunk(0)#[0]["data_row"]
#for one in chunk[:3]:
#    #print(one["metadata"])
#    row = one["data_row"]




SIMPLE = False

import torch
torch.cuda.empty_cache()


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
    # should play with HF trainer, which accepts options like
    # per_device_train_batch_size = 8,
    # gradient_accumulation_steps = 8 ~= batch 64 with less memory   

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=DATA_DIR,
        task_cache_base_path="./cache",
        train_task_name_list=TASK_NAMES.split(),
        val_task_name_list=TASK_NAMES.split(),
        train_batch_size=args.batch_size, # tony = 2!
        gradient_accumulation_steps =args.gradient_accumulation_steps, # à tester; équivalent à multiplier batch_size mais avec mémoire moindre
        #gradient_checkpointing=True, # TODO: not available but would be convenient to propagate to trnsformer trainer
        eval_batch_size=1,
        epochs=args.epochs,
        num_gpus=1,
    ).create_config()
    # manual additions ... should be handled externally too
    # 0 args: UniformMultiTaskSampler ProportionalMultiTaskSampler
    # 1 args: SpecifiedProbMultiTaskSampler "task_to_unweighted_probs" (dict)
    # 2 args: TemperatureMultiTaskSampler  <check noms: temperature + dict tache:nb exemples
    jiant_run_config["sampler_config"] = {"sampler_type": args.sampling_strategy}
    # sampling with more arguments: just added in the same dict
    # eg:
    # sampler_config = {
    #            "sampler_type": "SpecifiedProbMultiTaskSampler",
    #            "task_to_unweighted_probs": capped_num_examples_dict        
    # }
    # ---- saving 
    os.makedirs(os.path.join(EXP_DIR,"run_configs/"), exist_ok=True)
    # TODO: save under different names in each run logging directory
    OUTPUT_DIR = os.path.join("runs",RUN_NAME)
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    py_io.write_json(jiant_run_config,os.path.join(EXP_DIR,"run_configs/last_jiant_run_config.json"))
    py_io.write_json(jiant_run_config,os.path.join(OUTPUT_DIR,"run_config.json"))
    display.show_json(jiant_run_config)

    print("looking for model at : ",os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"))

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=os.path.join(EXP_DIR,"run_configs/last_jiant_run_config.json"),
        output_dir=OUTPUT_DIR,
        hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
        model_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"),
        model_config_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"/config.json"),
        learning_rate=1e-5, # tony = 1e-3
        eval_every_steps=args.eval_every_step,
        no_improvements_for_n_evals=args.no_improvements_for_n_evals,
        write_val_preds=True,
        fp16=args.fp16,
        do_train=True,
        do_val=True,
        do_save=True,
        force_overwrite=True,
    )
    # TODO: save the run config too (? already saved)

    main_runscript.run_loop(run_args)

    # predictions are stored only as torch tensors, this puts them in disrpt format
    # right now this only provably works for one task at a time
    # TODO: check pred file in multi-task exps
    # TODO: relax assertion that nb predictions = nb of tokens (when truncated sequence because too long for transformer)
    infile = os.path.join("runs",RUN_NAME,"val_preds.p")
    for one_task in task_list:
        try: 
            outfile = os.path.join("runs",RUN_NAME,one_task+"_dev.txt")
            convert_prediction_to_disrpt(infile,one_task,outfile,task_dev_caches[one_task])
        except:
            print("saved prediction not working with task:",one_task)

if CO2_tracking: tracker.stop()
