"""
Generate Disrpt segmented files with automated sentence split
from .tok files, given a path to the whole disrpt data (all corpus)
Also computes some stats about precision of sentence-beginning segment 
"""

# FIXME: 
# TODO
# - stats on sentence token nb distribution 
#     - tokens
#     - subtokens via transformer (more useful)
# x- generate split with ersatz
# x- generate split with stanza
# x- generate split with trankit (best)


from disrpt_io import SegmentCorpus, ConnectiveCorpus, get_language, TRANKIT_LANG_MAP

import argparse
import sys, os, os.path
from pathlib import PurePath




def compute_paths(corpus_path,corpus_name,sub):
    """given the sub-corpus path, its name and a split name for subcorpus train/test/dev, return input file and output file paths"""
    path = PurePath(corpus_path)
    # input file minus the suffix (will be tok and/or conllu)
    file_path = os.path.join(corpus_path,corpus_name+"_"+sub)#+".tok")
    basename = os.path.basename(file_path)
    lang = basename.split(".")[0][:2]
    output_path = os.path.join(corpus_path,corpus_name+"_"+sub)
    return file_path, output_path




def split_corpus(corpus_path,corpus_name,model="ersatz",suffix="",**kwargs):
    """
    apply the sentence splitter to all splits of a given subcorpus
    collect stats about precision of predicting segments at predicted sentence beginning

    kwargs might contain an open "pipeline" (eg. trankit pipeline) to pass on downstream for splitting sentences, so that it is not re-created for each paragraph
    """
    ok = True
    stats = []
    orig_lang, framework, name = corpus_name.split(".")
    lang = get_language(orig_lang,model)
    for sub in ["train","dev","test"]:
        file_path, output_path = compute_paths(corpus_path,corpus_name,sub)
        print(file_path, output_path)
        #print(file_path,output_path)
        if framework == "pdtb":
            corpus = ConnectiveCorpus()
        else:
            corpus = SegmentCorpus()
        print("doing ...",file_path)
        # [ok] TODO load both tok + connlu
        if os.path.exists(file_path+".tok"):#try: 
            corpus.from_file(file_path+".tok")
            corpus.sentence_split(model=model,lang=lang,**kwargs)
            tp, tot, all = corpus.eval_sentences()
            if suffix:
                final_name = f"{output_path}.{suffix}.split"
            else: 
                final_name = f"{output_path}.split"
            corpus.format(file=final_name)
            stats.append({"split":sub, "corpus":name, "framework":framework,"lang":orig_lang,
                        "src":"split",
                        "tp":tp,"total_sentences":tot,"total_segments":all})
        else:#except:
            print(f"WARNING, problem with file {file_path} (ignored)",file=sys.stderr)
    return ok, stats 


def sentence_stats(corpus_path,corpus_name):
    """
    produce stats about sentences from conllu format disrpt files

    TODO: factorize / previous function
    """
    ok = True
    stats = []
    orig_lang, framework, name = corpus_name.split(".")
    lang = get_language(orig_lang,None)
    for sub in ["train","dev","test"]:
        file_path, output_path = compute_paths(corpus_path,corpus_name,sub)
        print(file_path, output_path)
        #print(file_path,output_path)
        if framework == "pdtb":
            corpus = ConnectiveCorpus()
        else:
            corpus = SegmentCorpus()
        try:
            print("doing ...",file_path)
            # [ok] TODO load both tok + connlu
            corpus.from_file(file_path+".conllu")
            # TODO: add src and collect "ref" sentence stats from conllu
            tp, tot, all = corpus.eval_sentences(mode="conllu")
            stats.append({"split":sub, "corpus":name, "framework":framework,"lang":orig_lang,
                      "src":"conllu",
                      "tp":tp,"total_sentences":tot,"total_segments":all})
        except: 
            print("WARNING: pb computing stats for file (ignored)")
    return ok, stats 


def cmdline_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("corpus_path",
                help="path to a subcorpus directory")
    p.add_argument("--lang", default=None,help="corpus language (will be determined automatically if not specified)")
    p.add_argument("-m", "--model", choices=["ersatz","stanza","trankit","baseline"], default="trankit",
                help="sentence splitter model")
    p.add_argument("-o", "--output-path", default="sentence_split_stats.csv",
                help="file to save stats in")
    p.add_argument("--suffix",default="",help="add given suffix to each resulting file")
    p.add_argument("-s","--stats-conllu",default=False, action='store_true',help="only compute sentence stats on conllu corpora")
    p.add_argument("--gpu",default=False, action='store_true',help="use GPU")


    return(p.parse_args())

if __name__ == "__main__":
    from glob import glob 
    from tqdm import tqdm
    import pandas as pds
    import logging

    #logging.basicConfig(level=logging.DEBUG)
    os.environ["LOGLEVEL"] = "DEBUG"


    args = cmdline_args()

    kwargs = {}

    if args.model=="stanza":
        try:
            import stanza
            from stanza.pipeline.core import DownloadMethod
        except: 
            print("You need to install stanza",file=sys.stderr)
    elif args.model=="trankit":
        try:
            import trankit
        except: 
            print("You need to install trankit",file=sys.stderr)
    elif args.model=="ersatz":#ersatz
        try:
            import ersatz
            from disrpt_io import languages as ersatz_languages
        except: 
            print("You need to install ersatz",file=sys.stderr)
    else:
        pass

    corpus_path = args.corpus_path
    arg_lang = args.lang
    # read the dir content
    # TODO: check it is a dir
    subcorpora_path = [x for x in glob(os.path.join(corpus_path,'tur*')) if os.path.isdir(x)]
    #print(subcorpora_path)
    stats = []
    for subcorpus_path in subcorpora_path:
        # check that this is a subcorpus, whose name must be lang.framework.name
        subcorpus_name = os.path.basename(subcorpus_path)
        print(subcorpus_name)
        if not(len(subcorpus_name.split("."))==3):
            continue
        lang,framework,name = subcorpus_name.split(".")
        if args.model=="trankit" and not(args.stats_conllu):
            # needs to be initialized once only for performance
            lang = get_language(lang,args.model)
            if name=="gum" and lang=="english":
                lang = "english-gum"
            pipeline = trankit.Pipeline(lang,gpu=args.gpu)
            kwargs["pipeline"] = pipeline
        #--- the real code
        if not(args.stats_conllu):
            ok, res = split_corpus(subcorpus_path,subcorpus_name,model=args.model,suffix=args.suffix,**kwargs)
        #--- collect stats on sentence beginnings in conllu files
        else:
            ok, res = sentence_stats(subcorpus_path,subcorpus_name)
        stats.extend(res)

    # save stats about what was split / or stats about conllu for reference
    df = pds.DataFrame(stats)
    df.to_csv(args.output_path,index=False)