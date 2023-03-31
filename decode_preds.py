from logging import raiseExceptions
import torch
import sys, os.path
import argparse

"""reads tagging prediction tensor saved by jiant
might be easier to dump then at evaluation time but not general enough for inference

TODO: 
   x- refactor : this needs to call task specific formatting methods 
   x- plugs diapuncmentation tasks too
   x- more general : extract labels from task class
   x- check that the written predictions are indeed for each token, and not subtokens
   x- manage cached dataset to recover label_mask from tokeniczation
    - check that no errors from tokenization / mismatch nb of preds, unless sequence is > max_seq_length
    - generalise to train/val/test preds (just dev/validation for now)

"""
from jiant.tasks.evaluate.core import F1TaggingEvaluationScheme
from jiant.tasks.lib.disrpt21.disrpt21 import DisrptTask
from jiant.tasks.lib.disrpt21.disrpt21_connective import DisrptConnTask
from jiant.tasks.lib.diapuncmentation.punctuation_task import PunctuationTask
from jiant.tasks.lib.diapuncmentation.daseg_task import DaSegTask

import jiant.shared.caching as caching



def convert_prediction_to_disrpt(infile,task,outfile,cache,encoding="utf8"):
    """ infile: is the torch-saved tensor of predictions
        task: name of tasktask
        outfile: filename where to save predictions or "stdout"
    """
    #corpus_name = task.split("_")[1]
    corpus_name = task.split("_")[1] if "_" in task else ""
    if corpus_name in ("pdtb","tdb","cdtb") : 
        task_type = DisrptConnTask
    else: 
        task_type = DisrptTask

    orig_labels = task_type.ORIG_LABELS #for disrpt : ["_","BeginSeg=Yes"] ou ["_","Seg=B-Conn","Seg=I-Conn"]

    info_labels = F1TaggingEvaluationScheme.get_labels_from_cache_and_examples(task_type, cache,[])
    
    if task.startswith("disrpt21"):
        if corpus_name in ("pdtb","tdb","cdtb") : 
            task_type = DisrptConnTask
        else: 
            task_type = DisrptTask
    elif task.startswith("punctuation"):
        task_type = PunctuationTask
    elif task.startswith("daseg"):
        task_type = DaSegTask
    else:
        print("task not recognized")
        sys.exit(0)

    # contains prediction and label_mask
    data = torch.load(infile)[task]
    if outfile=="stdout":
        outfile = sys.stdout
    else:
        outfile = open(outfile,"w",encoding=encoding)

    # info_labels: contains reference label mask corresponding to sub-tokenized input, saved in the cache
    # necessary because the saved prediction tensor does not store explicitely the subtokenization 

    info_labels = F1TaggingEvaluationScheme.get_labels_from_cache_and_examples(task_type, cache,[])

    output = task_type.format_predictions(info_labels,data)
    print("\n".join(output),file=outfile)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("infile",help="result tensor path")
    parser.add_argument("task",help="name of the task")
    #parser.add_argument("--task-type",default="segment",help="task type segment or connective; connective automatically detected")
    parser.add_argument("--outfile",default="stdout",help="file to save results in, defaults to stdout")
    parser.add_argument("--encoding",default="utf8",help="output encoding")
    parser.add_argument("--cache-path",default="./cache",
                                    help="path to the general instances cache directory, defaults to ./cache")

    args = parser.parse_args()
    
    cache = caching.ChunkedFilesDataCache(os.path.join(args.cache_path,f"{args.task}/val"))
    convert_prediction_to_disrpt(args.infile,args.task,args.outfile,cache,encoding=args.encoding)
