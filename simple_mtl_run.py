# TODO
#    x- simple -> normal run
#     - config disrtp23 : conllu / split
#        FIXME : bug in saving config file for expe (name hardcoded in jiant/proj/simple/runscript.py (function create_and_write_task_configs)
#    /- external config
#    /- argparse for all arguments
#    - [should be priority] test/prediction mode == with-continue + no training
#    - refactor: 
#          this is a mess because of different training setups

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.proj.simple.runscript as simple_run
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
from decode_preds import convert_prediction_to_disrpt, get_last_run
#from datasets import load_dataset_builder
import os, sys
import glob
import argparse
from codecarbon import EmissionsTracker

EXP_DIR =  os.environ.get("EXP_DIR","/home/muller/Devel/jiant/exp")
DATA_DIR = os.path.join(EXP_DIR,"tasks","configs")

parser = argparse.ArgumentParser()

parser.add_argument("tasks",help="a string listing some tasks")
parser.add_argument("--run-name",default=None,help="name of the directory where to log experiments; if not set, will combine task and model")
parser.add_argument("--model-name",default="bert-base-multilingual-uncased",help="name of the base model to use")
parser.add_argument("--model-path",default=None,help="path to the model; if model-name is on hugging face, this does not need to be set; otherwise this is a local model, different from base model")
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
parser.add_argument("--test",action="store_true",default=False,help="do the final test evaluation")
# FIXME: not functional; todo this will be managed by predict.py 
parser.add_argument("--predict-only",action="store_true",default=False,help="prediction on val/test without training; model-path needs to be set and model-name must indicate the base model ")

#
parser.add_argument("--continue-with",default=None,help="start training with already existing fine-tuned model in argument, for sequential learning")

# TODO: write corresponding code in run below
# ood will cover two cases: 
#    - 0-shot eval for instance on the pdtb/tedm data
#    - merged corpora, where eval happens on the separate corpora, seen as "ood" tasks for the main model (although they were in the merged train)
#      list of ood will be list of tasks, eg disrpt23_tur_pdtb_tedm_conllu, with corresponding config files
parser.add_argument("--ood",default="",help="list of comma separated out-of-domain evaluation tasks; their config files need to exist and have no train;")
parser.add_argument("--batch-size",default=16,type=int,help="")
parser.add_argument("--gradient-accumulation-steps",default=4,type=int,help="delaying gradient update to allow for larger effective batches")
parser.add_argument("--epochs",default=1,type=float,help="nb of epochs for training")
parser.add_argument("--eval-every-step",default=100,type=int,help="")
parser.add_argument("--no_improvements_for_n_evals",default=5,type=int,
                    help="early stopping after n evals w/o improvements; needs eval-every step to be set")
parser.add_argument("--freeze-layers",default="",help = "freeze layers in the encoder, layers given as string: comma separated list of layers to freeze eg  1,2,3 or range 1-9, default=None")
parser.add_argument("--co2",action="store_true",default=False,help="track co2 emissions (needs internet access)")
parser.add_argument("--fp16",action="store_true",default=False,help="activate mixed precision 16/32bit; needs apex installed")

# dynamic learning rate: warmup_steps_proportion, eg 

#example model names: "bert-base-multilingual-uncased", "roberta-base"

args = parser.parse_args()
args_dict = vars(args)
kept_args = ["batch_size","epochs","gradient_accumulation_steps","max_seq_length","sampling-strategy","continue_with"]

hyper_params = {k:v for k,v in args_dict.items() if k in kept_args}

print(args)

EXP_DIR = args.exp_dir
#DATA_DIR = os.path.join(EXP_DIR,"tasks")
DATA_DIR = args.config_dir
TASK_NAMES = args.tasks
if args.model_path is None:
    HF_PRETRAINED_MODEL_NAME = args.model_name 
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME
else:# model specified = local (for prediction); TODO needs to be careful with names 
    HF_PRETRAINED_MODEL_NAME = args.model_name
    MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]

RUN_NAME = args.run_name if args.run_name is not None else f"run_{TASK_NAMES}_{MODEL_NAME}"

CO2_tracking = args.co2

if CO2_tracking: tracker = EmissionsTracker()

task_list = TASK_NAMES.split()

# testing the data reader
# TODO 0-shot: 0-shot tasks needs to be declared separately with phases = ["val"]
# TODO testing: ? phrases = test ?
if True: #not(args.predict_only):
    for task_name in task_list:
        #task_config_path=os.path.join(DATA_DIR,f"{task_name}_config.json")
        #print("config ?",task_config_path)
        tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
            task_config_path=os.path.join(DATA_DIR,f"{task_name}_config.json"),
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            output_dir=f"./cache/{task_name}",
            max_seq_length=args.max_seq_length,
            phases=["train", "val"]+ ([] if not(args.test) else ["test"]) ,
            smart_truncate = True,
        ))
    
    
if args.ood:
    print(f"ood tasks = {args.ood}",file=sys.stderr)
    args.ood = args.ood.split(",")
    # TODO: transform args.ood en liste
    for task_name in args.ood:
        tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
            task_config_path=os.path.join(DATA_DIR,f"{task_name}_config.json"),
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            output_dir=f"./cache/{task_name}",
            # [done] TODO: or test if in test mode (args.test)
            phases=["val"]+ ([] if not(args.test) else ["test"]),
            smart_truncate = True,
        ))
else:
    args.ood = []

task_dev_caches = {task_name:caching.ChunkedFilesDataCache(f"./cache/{task_name}/val") for task_name in task_list+args.ood}
if args.test:
    task_test_caches = {task_name:caching.ChunkedFilesDataCache(f"./cache/{task_name}/test") for task_name in task_list+args.ood}

#chunk = caching.ChunkedFilesDataCache("./cache/disrpt21_gum/train").load_chunk(0)#[0]["data_row"]
#for one in chunk[:3]:
#    #print(one["metadata"])
#    row = one["data_row"]




SIMPLE = False

import torch
torch.cuda.empty_cache()


if CO2_tracking: tracker.start()

# FIXME: not tested/not working (but cf with_continue below, which could be a basis for this)
if False:#args.predict_only: 
    args.test = True
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator( 
        task_config_base_path=DATA_DIR,
        task_cache_base_path="./cache",
        #train_task_name_list=TASK_NAMES.split(),
        val_task_name_list=TASK_NAMES.split(),
        test_task_name_list=TASK_NAMES.split(),
        train_batch_size=args.batch_size, # tony = 2!
        #gradient_accumulation_steps =args.gradient_accumulation_steps, # à tester; équivalent à multiplier batch_size mais avec mémoire moindre
        #gradient_checkpointing=True, # TODO: not available but would be convenient to propagate to trnsformer trainer
        eval_batch_size=1,
        epochs=args.epochs,
        num_gpus=1,
        ).create_config()
    os.makedirs("./run_configs/", exist_ok=True)
    py_io.write_json(jiant_run_config, "./run_configs/predict_run_config.json")
    display.show_json(jiant_run_config)

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path="./run_configs/predict_run_config.json",
        output_dir=os.path.join("runs",RUN_NAME),
        hf_pretrained_model_name_or_path="roberta-base",
        model_path=os.path.join(args.model_path,"best_model.p"), # Loading the best model
        model_load_mode="partial",
        model_config_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"/config.json"),
        #learning_rate=1e-5,
        #eval_every_steps=500,
        do_train=False,
        do_val=True,
        no_cuda=True,
        #force_overwrite=True,
        )
    main_runscript.run_loop(run_args)
else:
    # should play with HF trainer, which accepts options like
    # per_device_train_batch_size = 8,
    # [done] gradient_accumulation_steps = 8 ~= batch 64 with less memory   

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=DATA_DIR,
        task_cache_base_path="./cache",
        train_task_name_list=TASK_NAMES.split(),
        val_task_name_list=TASK_NAMES.split()+args.ood,
        # TODO:
        #    - for sequential learning or merged corpora, this should be adapted to the list of subtask
        test_task_name_list=(TASK_NAMES.split()+args.ood) if args.test else [],
        train_batch_size=args.batch_size, # tony = 2!
        gradient_accumulation_steps =args.gradient_accumulation_steps, # à tester; équivalent à multiplier batch_size mais avec mémoire moindre
        #gradient_checkpointing=True, # TODO: not available but would be convenient to propagate to trnsformer trainer
        eval_batch_size=8,
        epochs=args.epochs,
        num_gpus=1,
    ).create_config()
    ################# BUT ALSO THIS ########################"
    # make all tasks point to the trained head, which has TASK_NAMES in the config file (could be any name as long they are the same (?))
    if args.ood !=[]:
        for task_name in args.ood:
            jiant_run_config["taskmodels_config"]["task_to_taskmodel_map"][task_name] = TASK_NAMES
            print(f"ood task {task_name} pointed to model '{TASK_NAMES}'",file=sys.stderr)
    #os.makedirs("./run_configs/", exist_ok=True)
    #py_io.write_json(jiant_run_config, "./run_configs/jiant_run_config.json")
    
    
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
    # experimental additional meta_data
    # FIXME: should frozen layers be saved here to ? otherwise appear only in args.json
    jiant_run_config["specific_hyper_parameters"] = hyper_params
    os.makedirs(os.path.join(EXP_DIR,"run_configs/"), exist_ok=True)
    # TODO: save under different names in each run logging directory
    OUTPUT_DIR = os.path.join("runs",RUN_NAME)
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    py_io.write_json(jiant_run_config,os.path.join(EXP_DIR,"run_configs/last_jiant_run_config.json"))
    py_io.write_json(jiant_run_config,os.path.join(OUTPUT_DIR,"run_config.json"))
    display.show_json(jiant_run_config)

    if args.continue_with:
        print("continuing training from ",args.continue_with)
        run_args = main_runscript.RunConfiguration(
            jiant_task_container_config_path=os.path.join(EXP_DIR,"run_configs/last_jiant_run_config.json"),
            output_dir=OUTPUT_DIR,
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            # this is the already fine-tune model path
            model_path=os.path.join(args.continue_with,"best_model.p"),
            model_load_mode="partial",
            model_config_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"config.json"),
            learning_rate=1e-5, # tony = 1e-3
            freeze_layers = args.freeze_layers, # freeze layers in the encoder 
            eval_every_steps=args.eval_every_step,
            no_improvements_for_n_evals=args.no_improvements_for_n_evals,
            write_val_preds=not(args.predict_only),
            write_test_preds=True if args.test else False,
            fp16=args.fp16,
            # predict only = no train, no eval on dev either
            do_train=not(args.predict_only),
            do_val=not(args.predict_only),
            #keep_checkpoint_when_done=True,
            do_save=True,
            force_overwrite=True,
            )
        main_runscript.run_loop(run_args)

    else:
        print("looking for model at : ",os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"))

        run_args = main_runscript.RunConfiguration(
            jiant_task_container_config_path=os.path.join(EXP_DIR,"run_configs/last_jiant_run_config.json"),
            output_dir=OUTPUT_DIR,
            hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
            model_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"model/model.p"),
            model_config_path=os.path.join(EXP_DIR,"models",HF_PRETRAINED_MODEL_NAME,"config.json"),
            learning_rate=1e-5, # tony = 1e-3
            freeze_layers = args.freeze_layers, # freeze layers in the encoder 
            eval_every_steps=args.eval_every_step,
            no_improvements_for_n_evals=args.no_improvements_for_n_evals,
            write_val_preds=True,
            write_test_preds=True if args.test else False,
            fp16=args.fp16,
            do_train=True,
            do_val=True,
            #keep_checkpoint_when_done=True,
            do_save=True,
            force_overwrite=True,
        )


        main_runscript.run_loop(run_args)

    # predictions are stored only as torch tensors, this puts them in disrpt format
    # right now this only provably works for one task at a time
    # TODO: check pred file in multi-task exps
    # [done] TODO: relax assertion that nb predictions = nb of tokens (when truncated sequence because too long for transformer)
    if not(args.predict_only): 
        val_infile = os.path.join("runs",RUN_NAME,"val_preds.p")
    if args.test:
        test_infile = os.path.join("runs",RUN_NAME,"test_preds.p")
    for one_task in task_list+args.ood:
        if True: 
            if not(args.predict_only):
                outfile = os.path.join("runs",RUN_NAME,one_task+"_dev.txt")
                convert_prediction_to_disrpt(val_infile,one_task,outfile,task_dev_caches[one_task])
            if args.test:
                outfile = os.path.join("runs",RUN_NAME,one_task+"_test.txt")
                convert_prediction_to_disrpt(test_infile,one_task,outfile,task_test_caches[one_task])
        #except:
        #    print("saved prediction not working with task:",one_task)
        #-------------------
        # move a few files from the main task dir to the last run dir
        # best_model, config, metrics, prediction (dev)
        # [done] TODO: dev/test option   
        last_exp_dir = get_last_run(os.path.join("runs",RUN_NAME))
        files_to_move = []
        if not(args.predict_only):
            files_to_move.append(one_task+"_dev.txt")
        # test should be run only once, no need to move it for subruns, but it makes it easier to collect scores to treat them like dev
        if args.test:
            # ood tasks will be in their corresponding training corpus so just move every pred file
            files_to_move.append(one_task+"_test.txt")
        for one_file in files_to_move:
            os.replace(os.path.join("runs",RUN_NAME,one_file),os.path.join(last_exp_dir,one_file))
    # FIXME: does not work with multi-task default names   
    files_to_move = ["run_config.json","args.json"]+(["best_model.p","best_model.metadata.json",
                                                     "val_metrics.json","val_preds.p"] if not(args.predict_only) else [])+(["test_preds.p"] if args.test else [])
    for one_file in files_to_move:
        os.replace(os.path.join("runs",RUN_NAME,one_file),os.path.join(last_exp_dir,one_file))    

if CO2_tracking: tracker.stop()
