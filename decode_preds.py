import torch
import sys
import argparse

"""reads prediction tensor saved by jian
might be easier to dump then at evaluation time but not general enough for inference

   - todo ? more general : extract labels from task class
   - todo check that the written predictions are indeed for each token, and not subtokens
   

"""

def convert_prediction_to_disrpt(infile,task,outfile,encoding="utf8"):
    """ infile: is the torch-saved tensor of predictions
        task: name of task
        outfile: filename where to save predictions or "stdout"
    """
    if "pdtb" in task: 
        task_type = "connective"
    else: 
        task_type = "segment"

    if task_type == "segment":
        labels = ["_","BeginSeg=Yes"]
    else:
        labels = ["_","Seg=B-Conn","Seg=I-Conn"]
    
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
        outlabels = [labels[j] for j in preds[i]][:len(tokens)]
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

    args = parser.parse_args()
    
    convert_prediction_to_disrpt(args.infile,args.task,args.outfile,encoding=args.encoding)
