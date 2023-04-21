"""
functions to collect scores across experiments

normally jiant (PM version) create a dir for every experiment combining
corpus+conllu/split+model eg deu.rst.pcc / conllu / roberta base

inside each of these, there can be multiple runs saved in their own subdirs (named based on time). originally jiant saved only one set of args for that
but it's now modified to move prediction+metrics+config to each subdir



"""

import pandas as pds
import os.path
from glob import glob
import json

tasks = set(['deu_rst_pcc', 'eng_dep_scidtb', 'eng_rst_gum', 'eng_rst_rstdt', 'eng_sdrt_stac', 
 'eus_rst_ert', 'fas_rst_prstc', 'fra_sdrt_annodis',  'nld_rst_nldt', 
 'por_rst_cstn', 'rus_rst_rrt', 'spa_rst_rststb', 'spa_rst_sctb',  
 'zho_dep_scidtb',  'zho_rst_gcdt', 'zho_rst_sctb'])
pdtb_tasks = set(['eng_pdtb_pdtb','ita_pdtb_luna','tur_pdtb_tdb','zho_pdtb_cdtb',])

models = {
    "bert":"bert-base-multilingual-uncased",
    #"bert":"bert-base-multilingual-cased",
    "roberta": "xlm-roberta-base",
    "roberta-large": "xlm-roberta-large",
}

config = {"exp_dir":"../runs",
          "tasks" : tasks,
          "models": models,
         }

########################################################################################
####  various path manipulations according to conventions used in the main run script

def get_dir_from_taskname(taskname,task_type,model,campaign="disrpt23"):
    """get log dir name from task+task type + model """
    task = " ".join((f"{campaign}_{t}" for t in taskname.split()))
    return f"run_{task}_{task_type}_{model}"

# FIXME: will probably not work properly for Multi-task descriptions
def get_taskname_from_dir(dirname):
    """get jiant task description from log directory
    run_disrpt23_deu_rst_pcc_conllu_xlm-roberta-base -> disrpt23_deu_rst_pcc 
    """
    prefix, *taskname, task_type, model = dirname.split("_")
    return "_".join(taskname)

def get_taskname_from_description(desc):
    """convert jiant exp descriptions to simple corpus name
        disrpt23_deu_rst_pcc_conllu -> deu_rst_pcc
        FIXME: wont be ok for MT description
    """
    return ".".join(desc.split("_")[1:-1])

def retrieve_expe_dir(taskname,task_type,modelname,config):
    """retrieve the place where an experiment was logged
    follows the convention that a call to Multi-task joins task names with a space
    """
    model = config["models"][modelname]
    task_dir = get_dir_from_taskname(taskname,task_type,model)
    return os.path.join(config["exp_dir"],task_dir)

def get_all_run_dirs(path):
    """retrieve all directory name for a given config. """
    return [x for x in glob(os.path.join(path,"*")) if x.split(os.path.sep)[-1].isdigit()]


def get_last_run(path):
    """retrieve the directory name of the last experiment for a given config. 
    relies on the name of this being a number built with date+time so max = latest"""
    last = get_all_run_dirs(path)
    if last == []:
        return None
    return max(last)

#####################################################################
###  read / parse / collect various logs created by jiant

def get_log(path,logtype):
    """
    given path of an experiment log within a config dir, and a
    logtype = loss or metrics, indicating which type of data is retrieved
    either loss during training, or various metrics during training, on the validation set (p,r,f1)
    
    returns a dataframe with atomic info, either a loss for a training step for a task
    or an evaluation on validation set at a given step during training
    """
    mapping = {"loss":"loss_train.zlog",
               "metrics":"train_val.zlog"}
    data = pds.read_json(os.path.join(path,mapping[logtype]),lines=True)
    if logtype == "loss":
        # FIXME: not properly formatted, but loss is less useful for statistics collection so not urgent
        return data
    else:# should be factored out in another function
        all = dict(zip(["task","f1","precision","recall","task_step","global_step"],[[] for i in range(6)]))
        for idx, line in data.iterrows():
            metrics = line["metrics"]
            
            for task in metrics: 
                all["task"].append(get_taskname_from_description(task))
                for s in "f1","recall","precision":
                    all[s].append(float(metrics[task]["minor"][s]))
                all["global_step"].append(int(line["train_state"]["global_steps"]))  
                all["task_step"].append(int(line["train_state"]["task_steps"][task]))
        return pds.DataFrame(all)
    
    
def collect_log(tasks,models,logtype,task_type,config):
    """
    collects a set of experiments logs in one table, for either loss or validation metrics
    task_type is conllu or split
    logtype is loss or metrics
    eg 
        collect_results(["annodis","rststb","annodis rststb"],["roberta"],"metrics",config)
        collect_results(["annodis","rststb","annodis rststb"],["roberta"],"loss",config)
    """
    merged = []
    logs = {}
    for task in tasks:
        MTL = " " in task #MTL exp
        for m in models: 
            #try: 
            if True:
                path = get_last_run(retrieve_expe_dir(task,task_type,m,config))
                print(path)
                logs = get_log(path,logtype)
                #print(logs.columns)
                logs["setup"] = "single" if not(MTL) else "MTL: "+task
                logs["model"] = m
                logs["task_type"] = task_type
                merged.append(logs)
            #except:
            else:
                print("could not find data for",task,"+",m)
    return pds.concat(merged)


def collect_expe_results(tasks,task_type,models,config):
    """reads and merge loss and metrics learning curves into one table"""
    table = []
    for logtype in ("metrics","loss"):
        table.append(collect_log(tasks,models,logtype,task_type,config))
   
    all = pds.merge(*table,"outer")
    return all


# read metrics for each sub-experiment ... this was changed from case where only the one was saved so wip
# then will should be called on all "numbered" dir containing one run
def read_final_metrics(path,dataset="val",best_metadata=False):
    if best_metadata: 
        filename = "best_model.metadata.json"
    else:
        filename = "%s_metrics.json"%dataset
    filepath = os.path.join(path,filename)
    if not(os.path.exists(filepath)):
        return pds.DataFrame()
    data = json.load(open(filepath))
    #print(data)
    results = []
    if best_metadata: data = data["val_state"]["metrics"]
    params = read_run_config(path,keep=["freeze_layers"])
    for key in data: 
        if key!="aggregated":
            task = key
            metrics = []
            for one in ("precision","recall","f1"):
                if best_metadata: 
                    metrics.append(data[task]["minor"][one])
                else:
                    metrics.append(data[task]["metrics"]["minor"][one])
            *task, task_type = task.split("_",1)[1].split("_")
            results.append(["_".join(task),task_type]+metrics+list(params.values()))
    return pds.DataFrame(results,columns=["task","task_type","precision","recall","f1"]+list(params.keys()))

def read_run_config(path,keep=["freeze_layers"]):
    filename = "args.json"
    data = json.load(open(os.path.join(path,filename)))
    return {x:data[x] for x in keep}
 
def test_get_all_runs(path):
    """path points to a run with subexperiments"""
    all_runs = get_all_run_dirs(path)
    for run in all_runs:
        print(run)
        print(read_final_metrics(run,best_metadata=False))
        
        
def collect_final_result(tasks,models,config,task_type="conllu",dataset="val",best_metadata=False):
    """collect final results for best model for latest experiment on a collection of task X models"""         
    merged = []
    for task in tasks:
        MTL = " " in task #MTL exp
        for m in models: 
            if True:
            #try: 
                path = retrieve_expe_dir(task,task_type,m,config)
                all_runs = get_all_run_dirs(path)
                for run in all_runs:
                    scores = read_final_metrics(run,dataset=dataset,best_metadata=best_metadata)
                    scores["setup"] = "single" if not(MTL) else "MTL: "+task
                    scores["model"] = m
                    merged.append(scores)
            else:
            #except:
                print("could not find data for",task,"+",m)
            
    return pds.concat(merged)




# testing
if __name__=="__main__":
    # config contains all info to run this: mapping of tasks / main run directory
    val_scores = collect_final_result(config["tasks"],["roberta","roberta-large"],config,best_metadata=False)
    
    # various tests
    TESTING = False
    TEST_TASK = "deu_rst_pcc"

    if TESTING: 
        test_dir = retrieve_expe_dir(TEST_TASK,"conllu","roberta",config)
        print(test_dir)
        print(get_taskname_from_dir(test_dir))
        path = get_last_run(test_dir)
        log = get_log(path,"metrics")
        df = collect_expe_results(["deu_rst_pcc"],"conllu",["roberta","roberta-large"],config)
        
        # Example visualizations of loss/validation curves
        #alt.data_transformers.disable_max_rows()
        filter = (df["task"]=="deu.rst.pcc") & (df["model"]=="roberta-large") #& (df["global_step"]<5000)
        df[filter]["f1"].plot()
        
        path = retrieve_expe_dir("deu_rst_pcc","conllu","roberta-large",config)
        all_runs = get_all_run_dirs(path)
        for run in all_runs:
            print(run)
            print(read_final_metrics(run,best_metadata=False))