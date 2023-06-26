### this script is for disrpt task 1+2 2023: 
### evaluate predictions from separate files put in same dir
### report scores in one table (csv)


### Author : Laura Riviere

# TODO:
#  - use os.path.join for robust file pointing
#  - add option split/conllu
#  - find automatically all corpora in gold dir

import subprocess as subp
import argparse
import seg_eval 
import os, re
from glob import glob
import pandas as pd
from datetime import datetime


"""
2 situations : plain text (.toks), syntactically parsed text (.conllu)
"""

CORPORA_all = ["deu.rst.pcc","eng.dep.covdtb","eng.dep.scidtb","eng.pdtb.pdtb","eng.pdtb.tedm","eng.rst.gum",
               "eng.rst.rstdt","eng.sdrt.stac","eus.rst.ert","fas.rst.prstc","fra.sdrt.annodis","ita.pdtb.luna",
               "nld.rst.nldt","por.pdtb.crpc","por.pdtb.tedm","por.rst.cstn","rus.rst.rrt","spa.rst.rststb",
               "spa.rst.sctb","tha.pdtb.tdtb","tur.pdtb.tdb","tur.pdtb.tedm","zho.dep.scidtb","zho.pdtb.cdtb",
               "zho.rst.gcdt","zho.rst.sctb"]
extras = [f"fra.sdrt.cid{i+1}" for i in range(8)]
#CORPORA_all = ["deu.rst.pcc","eng.dep.covdtb","eng.pdtb.pdtb","eng.pdtb.tedm","eus.rst.ert","fra.sdrt.annodis","nld.rst.nldt","por.pdtb.crpc","por.pdtb.tedm","rus.rst.rrt","spa.rst.rststb","spa.rst.sctb","tha.pdtb.tdtb","tur.pdtb.tdb","tur.pdtb.tedm","zho.dep.scidtb","zho.pdtb.cdtb","zho.rst.gcdt","zho.rst.sctb"]
corpora_pb = ["eng.dep.scidtb","eng.rst.gum","eng.rst.rstdt","eng.sdrt.stac","fas.rst.prstc","ita.pdtb.luna","por.rst.cstn"]

CORPORA_t1 = []
CORPORA_t2 = []

def get_stamp():
    now = datetime.now()
    stamp = re.sub('[\s:]', '_', str(now))
    return stamp

class Evaluation:
    def __init__(self, g:str, p:str,outdir=".",task_type="conllu"):
        self.g_dir = g
        self.p_dir = p
        self.outdir = outdir
        self.task_type = task_type 
        self.run = self.p_dir.split("/")[-2]
        print(f"---------------{self.run}")
        
    def evaluate(self,target="dev"): # normalize = lang.framework.name_part.format.pred
        fields = ['doc_name', 'tok_count', 'seg_type', 'gold_seg_count', 'pred_seg_count', 'prec', 'rec', 'f_score']
        
        self.conllu_scores = pd.DataFrame(columns = fields)

        i = 0
        for corpus in CORPORA_all+extras:
            i += 1
            print(f">>> {i} process {corpus}")
            pred = f"{corpus}_{target}.{self.task_type}.pred"
            pred_path = f"{self.p_dir}/{pred}"
            print(pred_path)
            if os.path.isfile(pred_path):
                if self.task_type=="split":
                    suffix = "tok"
                else:
                    suffix = "conllu"
                gold_path = f"{self.g_dir}/{corpus}/{corpus}_{target}.{suffix}"
                scores_dict = seg_eval.get_scores(gold_path, pred_path)
                print(scores_dict)
                self.conllu_scores.loc[len(self.conllu_scores)] = [scores_dict[key] for key in fields]
        print(self.conllu_scores)

    def print(self,target="dev"):
        out_dir = self.outdir
        file_name = f"{self.run}_scores_segmentation_{target}.csv"
        path = f"{out_dir}{file_name}"
        if os.path.isfile(f"{out_dir}{file_name}"):
            path = f"{out_dir}{get_stamp()}_{file_name}"

        self.conllu_scores.to_csv(path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("golddir", help="gold directory")
    p.add_argument("preddir", help="predictions directory") # data_clean_all
    p.add_argument("--outdir",default="scores",help="destination dir for storing score files")
    p.add_argument("--task-type",default="conllu",choices=["conllu","tok","split"],help="conllu ou tok/split")

    
    o = p.parse_args()
    if o.task_type == "tok": o.task_type = "split"
    
    ev = Evaluation(o.golddir, o.preddir,outdir=o.outdir,task_type=o.task_type)
    
    for target in ["dev","test"]:
        ev.evaluate(target=target)
        ev.print(target=target)
