from logging import raiseExceptions
import torch
import sys, os.path
import argparse

"""reads tagging prediction tensor saved by jiant
might be easier to dump then at evaluation time but not general enough for inference

   x- todo ? more general : extract labels from task class
   x- todo check that the written predictions are indeed for each token, and not subtokens
    - manage cached dataset to recover label_mask from tokenization
    - generalise to train/val/test preds (just dev/validation for now)

"""
from jiant.tasks.evaluate.core import F1TaggingEvaluationScheme
from jiant.tasks.lib.disrpt21.disrpt21 import DisrptTask
from jiant.tasks.lib.disrpt21.disrpt21_connective import DisrptConnTask
import jiant.shared.caching as caching



def convert_prediction_to_disrpt(infile,task,outfile,cache,encoding="utf8"):
    """ infile: is the torch-saved tensor of predictions
        task: name of tasktask
        outfile: filename where to save predictions or "stdout"
    """
<<<<<<< Updated upstream
    corpus_name = task.split("_")[1]
    if corpus_name in ("pdtb","tdb","cdtb") : 
        task_type = DisrptConnTask
    else: 
        task_type = DisrptTask

    orig_labels = task_type.ORIG_LABELS #for disrpt : ["_","BeginSeg=Yes"] ou ["_","Seg=B-Conn","Seg=I-Conn"]

    info_labels = F1TaggingEvaluationScheme.get_labels_from_cache_and_examples(task_type, cache,[])
=======
    corpus_name = task.split("_")[1] if "_" in task else ""
>>>>>>> Stashed changes
    

    data = torch.load(infile)[task]

    meta = data["meta"]
    preds = data["preds"]

    if outfile=="stdout":
        outfile = sys.stdout
    else:
        outfile = open(outfile,"w",encoding=encoding)

    for i,instance in enumerate(meta):
        doc_id, sentence_id, sent_string = instance
        if doc_id!="": 
            print(f"# doc_id: {doc_id}",file=outfile)
        print(f"# sentence_id: {sentence_id}",file=outfile)
        print(f"# text = {sent_string}",file=outfile)
        tokens = sent_string.split()
        # this is were the label_mask should be used
        label_mask = info_labels[i]["label_mask"]
        relevant_preds = preds[i][label_mask]
        outlabels = [orig_labels[j] for j in relevant_preds]
        try: 
            assert len(outlabels)==len(tokens)
        except AssertionError:
            print(f"docid={doc_id},sentence_id={sentence_id},sent={sent_string}")
            print(f"out={len(outlabels)},tokens={len(tokens)}")
            raise AssertionError
        for n,tok in enumerate(tokens):
            # disrpt format (no feature)
            # token_id token _ _ _ _ _ _ _ label
            print(f"{n+1}\t{tok}\t"+"\t".join(["_"]*7)+f"\t{outlabels[n]}",file=outfile)
        print(file=outfile)


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
