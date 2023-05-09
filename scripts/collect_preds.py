from collect_scores import copy_all_predictions, config, tasks, pdtb_tasks, ood_tasks
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir",help="experiment log directory")#default="./runs"
    parser.add_argument("outdir",help="destination directory (must exist)")#default="/moredata/disrpt/predictions/test"
    parser.add_argument("--task_type",default="conllu",choices = ["conllu","tok","split"],help="conllu or tok(split)")
    parser.add_argument("--tasks",default = (tasks | pdtb_tasks), help="list of tasks or all (default all but ood)")
    parser.add_argument("--model",default = "roberta", choices = ["bert", "roberta","roberta-large"], help="model (bert, roberta or roberta-large)")
    parser.add_argument("--target",default = "dev", choices = ["dev","val","test"],help="dev(val) or test")

    args = parser.parse_args()
    
    
    copy_all_predictions(args.outdir,args.tasks,args.model,config,task_type=args.task_type,target=args.target)