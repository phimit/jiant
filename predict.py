# Test local trained model

# NB: 
# - Huggingface cant load robertalarge for sequence classification, which would be enough for the non-multi-task version
# - most general case: do a no-train + test eval
#          x- initialize a model exactly like continue training but with no training
#          x- get in eval mode
#          x- preprocess the input -> pb is can we ignore : the cache ? the test location in task config (bypass with temp file) ? 
#          x- predict
#           - convert prediction 
#           - parameters:
#                  task specification (pdtb or seg or else)
#           - refactor as a class
# tricky to make it simpler cos there are too many levels of abstraction + input preprocessing is normally cached beforehand
# inverstigate: debug in eval mode to see what kind of object the model really is, and if prediction works as expected

# TODO
#   - test a running pipe-line: simple text input or file -> disrpt format prediction file 
#        check: 
#            [ok]- tokenization
#                   x generate local config
#            [ok]- config generation
#            [ok]- prediction
#            [ok]- format conversion
#            - cleanup
#            - check model class while in debug mode
#   - change output:
#        - convert to brackets ?
#        - more generic format: output segment spans ? (eg for argilla ?)
#   - factor elements -> class + library file
#   - script using just factored library
#   - (more generally): update jiant setup to make a proper module 
#   - beta testing as seg user 


# FIXME 
#      - saved predictions do not match prediction by the model when called via the main script and no_train, why ? 
#        need to control/check: 
#           - what happens if the exact same config files are used instead of generating dummies ?
#                -> wont work cos task name is changed 
#                a more appropriate mode would be to map to task names, using the pretrained model name / description
#                 eg disrpt23_eng_rst_gum_conllu -> task disrpt23_eng_rst_gum_conllu -> point to task config and just change the pointer to the test file
#                 what about the run_script config ? could be copied if task name are correctly present
#           - are the proper weights used ? compare when loading


from string import Template
import os.path

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
#import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
#import jiant.proj.simple.runscript as simple_run
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
#import jiant.utils.display as display

from decode_preds import convert_prediction_to_disrpt
#from prediction import default_config, create_config

# if sentences not given
import trankit 
#import scripts.disrpt_io
import tempfile
  

# TODO: wrap it all in a class

class Predictor: 

    _default_config = {}
    
    # jiant config keys changed to be more intuitive, this maps back to the expected keys
    _map_keys = {"base_model":"hf_pretrained_model_name_or_path",
                 "segmentation_model":"model_path",
                 }

    def __init__(self,**config):
        """config should have the following keys (with examples):

        "base_model":"xlm-roberta-base", -> base encoder
        "model_dir":"/home/myname/jiant/models",  -> base encoders local download directory; can contain more than one model
        "segmentation_model":""/home/myname/jiant/pretrained/englishrst_rstdt", # path to fine-tuned model
        "output_dir":"/home/myname/temp_output", # where to put everything created; will be created if non existent
        "main_dir":"/home/myname/jiant/exp" # where are resources: models + run_configs + cache  (mut exist)
        """
        self.config = self._default_config.copy()
        self.config.update(config)
    

# basic run options that should not be changed for prediction mode
_default_config = {
            "model_load_mode":"partial",
            "write_val_preds":False,
            "freeze_layers":"0-23",
            "write_test_preds":True,
            "do_train":False,
            "do_val":False,
            "do_save":True,
            "force_overwrite":True
            }

def create_doc(document,output_dir):
    """
        document: str
        output_dir: str
        returns: str, path to the input file
    """
    # TODO
    #  - document is list of tokenized sentences
    #  - check if output_dir exists
    #  - check if output_dir is a directory
    #  - proper tokenization
    #  - manage list of sentences
    temp = tempfile.NamedTemporaryFile(mode="w",dir=output_dir,delete=False)
    header = ["# doc id = prediction"]
    tokens = document.split()
    # disrpt format has 10 fields + label, here set by default to "_"
    # TODO: add a way to change the default label
    lines = ["\t".join([str(i+1),x]+["_"]*8) for (i,x) in enumerate(tokens)]
    temp.write("\n".join(header+lines))
    temp.close()
    return temp.name


def create_config(name, 
                  input_file,
                  output_dir,
                  config_file_path,
                  cfg_template="resources/cfg_template.json"):
    """   
        name: str to name the "task"
        input_file: path to where input is going to be stored
        output_dir: where the cfg file will be stored (needs to exist)
        config_file_name: output cfg file name (eg local_cfg.json)
        
        returns: None
    """
    cfg_path = config_file_path
    new_config = open(cfg_path,"w")
    cfg = Template(open(cfg_template).read())
    output = cfg.substitute({"input_file":os.path.join(output_dir,input_file)})
    new_config.write(output)
    new_config.close()
    #return cfg_path
    

CLEAN_UP = False
# should be read from a file / arguments of the main script
document = "This is an unbelievable test. Now we will make another one."
# from gum test; with gum / conllu / roberta large, segments at "In" and "concerning" and "." 
document = "In many respects , researchers already possess a wealth of knowledge concerning the origins and consequences of discrimination and bias . [ 11 ]"
lang = "english"
name = "my_preds"
########## if we need to presegment in sentences ##########
# not necessary if right name chosen from the start, or trankit auto mode (slower)
#lang = disrpt_io.get_lang(lang,"trankit")
#pipeline = trankit.Pipeline(lang,gpu=0)
#trk_sentences = pipeline.ssplit(document)
#sentences = [s["text"] for s in trk_sentences["sentences"]]



##################################### this should be in main ##########
import copy
#config = _default_config.copy()
config = {
    "base_model":"xlm-roberta-large",
    "segmentation_model":"pretrained/english_rst_gum", # conllu by default
    "output_dir":"output_dir", # create if not existing
    "main_dir":"/home/muller/Devel/jiant"
    }

os.makedirs(config["output_dir"],exist_ok=True)
local_cache = os.path.join(config["output_dir"],"local")
os.makedirs(local_cache,exist_ok=True)

############# this should be factored out ###############
# 1) save input doc as disrpt format in output-dir (tokenized with ?) ; should we add a different read for simple inputs ?
# TODO
input_file = create_doc(document,config["output_dir"])
# 2) generate local config pointing to that file (TODO)
local_cfg_path = os.path.join(config["output_dir"],"local_config.json")
create_config(name,input_file,config["output_dir"],local_cfg_path)
# 3) tokenize resulting file
tokenize_cfg = {
        "task_config_path":local_cfg_path,
        "hf_pretrained_model_name_or_path":config["base_model"],
        "output_dir":local_cache,
        "max_seq_length":180,
        "phases": ["test"],
        "smart_truncate" : True,
        }

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(**tokenize_cfg))
# tokenize and save the result (TODO)
# ?? tokenize_and_save(document,config["output_dir"])
test_cache = caching.ChunkedFilesDataCache(os.path.join(tokenize_cfg["output_dir"],"test"))
##########################################
# [OK] test is properly done ? 
chunk = test_cache.load_chunk(0)#[0]["data_row"]
#for one in chunk[:3]:
#     print(one["metadata"])
#     row = one["data_row"]
############################# ok up to here #####################

# generate run config and save it
# 
#DATA_DIR = os.path.join(config["main_dir"],"exp","tasks","configs")

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=os.path.join(config["output_dir"]),
        task_cache_base_path=config["output_dir"],
        #train_task_name_list=TASK_NAMES.split(),
        #val_task_name_list=TASK_NAMES.split()+args.ood,
        test_task_name_list="local",
        train_batch_size=1, 
        #gradient_accumulation_steps =args.gradient_accumulation_steps, # à tester; équivalent à multiplier batch_size mais avec mémoire moindre
        #gradient_checkpointing=True, # TODO: not available but would be convenient to propagate to trnsformer trainer
        eval_batch_size=8,
        epochs=1,
        num_gpus=1,
    ).create_config()

py_io.write_json(jiant_run_config,os.path.join(config["output_dir"],"run_config.json"))


# prepare additional 
run_config = {
            "jiant_task_container_config_path":os.path.join(config["output_dir"],"run_config.json"),
            "hf_pretrained_model_name_or_path":config["base_model"],
            # this is the already fine-tune model path
            "model_path":os.path.join(config["segmentation_model"],"best_model.p"),
            "model_load_mode":"partial",
            "model_config_path":os.path.join(config["main_dir"],"models",config["base_model"],"config.json"),
            "output_dir": config["output_dir"]
    }
run_config.update(_default_config)

# this should be factored too
run_args = main_runscript.RunConfiguration(**run_config)
main_runscript.run_loop(run_args)

# conversions: predictions are in test_preds 
test_infile = os.path.join(config["output_dir"],"test_preds.p")
outfile = os.path.join(config["output_dir"],"local_test.txt")
# force task to be regarded as segmentation (disrpt_xx_seg) or pdtb (disrpt_xx_pdtb)
convert_prediction_to_disrpt(test_infile,"disrpt_xx_seg",outfile,test_cache,force_task_name="local")

# clean-up temp file
if CLEAN_UP:
    # remove output-dir
    pass
