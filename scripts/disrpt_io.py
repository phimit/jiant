"""
Classes to read/write disrpt-like files
+ analysis of sentence splitter / "gold" sentences or stanza/spacy sentences
    - ersatz

Disrpt is a discourse analysis campaign with (as of 2023): 
 - discourse segmentation information, in a conll-like format
 - discourse connective information (also conll-like)
 - discourse relations pairs, in a specific format

data are separated by corpora and language with conventionnal names
as language.framework.corpusname
eg fra.srdt.annodis

TODO: 
   - refactor how sentences are stored with dictionary: "connlu" / "tok" / "split"
        [ok] dictionary
        ? refactor creation of corpus/documents to allow for update (or load tok+conllu at once)
   - [ok] italian luna corpus has different meta tags avec un niveau supplémentaire: newdoc_id/newturn_id/newutterance_id
   - [ok] check behaviour on languages without pretrained models/what candidates ? 
        - nl, pt, it -> en?  
        - thai -> multilingual
   - test different candidates sets for splitting locations:    
        - [done] all -> trop sous-spécifié et trop lent 
        - [ok] en on all but zho+thai
        - (done] en à la place de multilingual ?
            bad scores on zho
    - [ok] fix bad characters: BOM, replacement char etc
            spécial char for apostrophe, cf
            data_clean/eng.dep.scidtb/eng.dep.scidtb_train.tok / newdoc_id = P16-1030 prob de char pour possessif
            ��antagonist��

            pb basque: "Osasun-zientzietako Ikertzaileen II ." nb tokens ...
                Iru�eko etc
    - pb turk: tur.pdtb.tdb/tur.pdtb.tdb_train: BOM ? '\ufeff' -> 'Makale'
            + extra blanc dans train (785)? 
            774	olduğunu	_	_	_	_	_	_	_	_
            775	söylüyor	_	_	_	_	_	_	_	_
            776	:	_	_	_	_	_	_	_	_
            777	Türkiye	_	_	_	_	_	_	_	_
            778	demokrasi	_	_	_	_	_	_	_	_
            779	istiyor	_	_	_	_	_	_	_	_
            780	ÖDPGenel	_	_	_	_	_	_	_	_
            781	Başkanı	_	_	_	_	_	_	_	_
            782	Ufuk	_	_	_	_	_	_	_	_
            783	Uras'tan	_	_	_	_	_	_	_	_
            784	:	_	_	_	_	_	_	_	_
            785		_	_	_	_	_	_	_	_
            786	Türkiye	_	_	_	_	_	_	_	_
            787	,	_	_	_	_	_	_	_	_
            788	AİHM'de	_	_
    - pb zh
        zh: ？是 is this "?"  listed in ersatz ? 
        ??hosto2
        sctb 3.巴斯克
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
   - specific preproc: 
        annodis/gum: titles
        gum/rrt : biblio / articles
        scidtb ?
   - different sentence splitters
        - [ok] ersatz
        - trankit
        - [abandoned] stanza: FIXME: lots of errors done by stanza eg split within words (might be due to bad input tokenization) 
   - [done] write doc in disrt format (after transformation for instance)
   - [done] eval of beginning of sentences (precision)
   - [done] (done in split_sentence script) eval / nb sentences connl  ~= recall sentences
   - eval length sentences (max)
   - [moot] clean main script : arguments/argparse -> script à part
   - [done] method for sentence splitting (for tok)
   - [done] iterate all docs in corpus
   - [done] choose language according to corpus name automatically
   - ?method for sentence resplitting for conllu ? needs ways of indexing tokens for later reeval ? or eval script does not care ?

   
candidate sets for splitting: 

    - multilingual (default) is as described in ersatz paper == [EOS punctuation][!number]
    - en requires a space following punctuation
    - all: a space between any two characters
    - custom can be written that uses the determiner.Split() base class



"""
import sys, os
import dataclasses
from itertools import chain
from collections import Counter
from copy import copy, deepcopy
from tqdm import tqdm
#import progressbar
#from ersatz import split, utils
import trankit
#import stanza
#from stanza.pipeline.core import DownloadMethod

# needed to track the mistakes made in preprocessing of the disrpt dataset, whose origin is unknown
BOM = '\ufeff'
REPL_CHAR = "\ufffd" # �

test_doc_seg = """# newdoc id = geop_3_space
1	La	le	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	2	det	_	BeginSeg=Yes
2	Space	space	PROPN	_	_	0	root	_	_
3	Launcher	Launcher	PROPN	_	_	2	flat:name	_	_
4	Initiative	initiative	PROPN	_	_	2	flat:name	_	_
5	.	.	PUNCT	_	_	2	punct	_	_

1	Le	le	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	2	det	_	BeginSeg=Yes
2	programme	programme	NOUN	_	Gender=Masc|Number=Sing	10	nsubj	_	_
3	de	de	ADP	_	_	4	case	_	_
4	Space	space	PROPN	_	_	2	nmod	_	_
5	Launcher	Launcher	PROPN	_	_	4	flat:name	_	_
6	Initiative	initiative	PROPN	_	_	4	flat:name	_	_
7	(	(	PUNCT	_	_	8	punct	_	BeginSeg=Yes
8	SLI	SLI	PROPN	_	_	4	appos	_	_
9	)	)	PUNCT	_	_	8	punct	_	_
10	vise	viser	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	BeginSeg=Yes
11	à	à	ADP	_	_	12	mark	_	_
12	développer	développer	VERB	_	VerbForm=Inf	10	ccomp	_	_
13	un	un	DET	_	Definite=Ind|Gender=Masc|Number=Sing|PronType=Art	14	det	_	_
14	système	système	NOUN	_	Gender=Masc|Number=Sing	12	obj	_	_
15	de	de	ADP	_	_	16	case	_	_
16	lanceur	lanceur	NOUN	_	Gender=Masc|Number=Sing	14	nmod	_	_
17	réutilisable	réutilisable	ADJ	_	Gender=Masc|Number=Sing	16	amod	_	_
18	entièrement	entièrement	ADV	_	_	19	advmod	_	_
19	inédit	inédit	ADJ	_	Gender=Masc|Number=Sing	14	amod	_	_
20	.	.	PUNCT	_	_	10	punct	_	_

# newdoc id = ling_fuchs_section2
1	Théorie	théorie	PROPN	_	_	0	root	_	BeginSeg=Yes
2	psychomécanique	psychomécanique	ADJ	_	Gender=Masc|Number=Sing	1	amod	_	_
3	et	et	CCONJ	_	_	4	cc	_	_
4	cognition	cognition	NOUN	_	Gender=Fem|Number=Sing	1	conj	_	_
5	.	.	PUNCT	_	_	1	punct	_	_
"""

# token is just a simple record type 
Token = dataclasses.make_dataclass("Token","id form lemma pos xpos morph head_id dep_type extra label".split(),
                                   namespace={'__repr__': lambda self: self.form,
                                              'format': lambda self: ("\t".join(map(str,dataclasses.astuple(self)))),
                                              # ignored for now cos we just get rid of MWE when reading disrpt file
                                              # but could be changed in the future
                                              #'is_MWE': lambda self: type(self.id) is str and "-" in self.id,
                                              }
                                   )


class Sentence:

    def __init__(self,token_list,meta):
        self.toks = token_list
        self.meta = meta

    def __iter__(self):
        return iter(self.toks)
    
    def __len__(self):
        return len(self.toks)

    def display(self,segment=False):
        """if segment option set to true, print sentences with marking of EDUs"""
        if segment: 
            output = [f"{'|' if token.label=='BeginSeg=Yes' else ''}{token.form}" for token in self]
            return " ".join(output)+"|"
        else:
            return self.meta["text"]

    def __in__(self,word):
        for token in self.toks:
            if token.form == word:
                return True
        return False

    def __repr__(self):
        return self.display()   

    def format(self):
        meta = f"# sent_id = {self.meta['sent_id']}\n" + f"# text = {self.meta['text']}\n"
        output = "\n".join([t.format() for t in self.toks])
        return meta+output

# not necessary because of trankit auto-mode but probably safer at some point
# why dont they use normalized language codes !!??
TRANKIT_LANG_MAP = {
    "de": "german",
    "en":"english",
    # to be tested
    "gum": "english-gum",
    "fr":"french",
    "it": "italian",
    "sp": "spanish",
    "es": "spanish",
    "eu": "basque",
    "zh": "chinese",
    "ru": "russian",
    "tr": "turkish",
    "pt":"portuguese",
    "fa": "persian", 
    "nl":"dutch",
    # blah
}

lg_map = {"sp":"es",
          "po":"pt",
          "tu":"tr"}


def get_language(lang,model):
    lang = lang[:2]
    if lang in lg_map:
        lang = lg_map[lang]
    if model=="ersatz":
        if lang not in ersatz_languages:
            lang = "default-multilingual"
    if model=="trankit":
        lang = TRANKIT_LANG_MAP.get(lang,"auto")
    return lang

# This is taken from ersatz https://github.com/rewicks/ersatz/blob/master/ersatz/candidates.py
# sentence ending punctuation
# U+0964  ।   Po  DEVANAGARI DANDA
# U+061F  ؟   Po  ARABIC QUESTION MARK
# U+002E  .   Po  FULL STOP
# U+3002  。  Po  IDEOGRAPHIC FULL STOP
# U+0021  !   Po  EXCLAMATION MARK
# U+06D4  ۔   Po  ARABIC FULL STOP
# U+17D4  ។   Po  KHMER SIGN KHAN
# U+003F  ?   Po  QUESTION MARK
# U+2026 ...  Po  Ellipsis
# U+30FB 
# U+002A *

# other acceptable punctuation
# U+3011  】  Pe  RIGHT BLACK LENTICULAR BRACKET
# U+00BB  »   Pf  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
# U+201D  "   Pf  RIGHT DOUBLE QUOTATION MARK
# U+300F  』  Pe  RIGHT WHITE CORNER BRACKET
# U+2018  ‘   Pi  LEFT SINGLE QUOTATION MARK
# U+0022  "   Po  QUOTATION MARK
# U+300D  」  Pe  RIGHT CORNER BRACKET
# U+201C  "   Pi  LEFT DOUBLE QUOTATION MARK
# U+0027  '   Po  APOSTROPHE
# U+2019  ’   Pf  RIGHT SINGLE QUOTATION MARK
# U+0029  )   Pe  RIGHT PARENTHESIS

ending_punc = {
    '\u0964',
    '\u061F',
    '\u002E',
    '\u3002',
    '\u0021',
    '\u06D4',
    '\u17D4',
    '\u003F',
    '\uFF61',
    '\uFF0E',
    '\u2026',
}

closing_punc = {
    '\u3011',
    '\u00BB',
    '\u201D',
    '\u300F',
    '\u2018',
    '\u0022',
    '\u300D',
    '\u201C',
    '\u0027',
    '\u2019',
    '\u0029'
}

list_set = {
    '\u30fb',
    '\uFF65',
    '\u002a', # asterisk
    '\u002d',
    '\u4e00' 
}
class Document:
    _hard_punct = {"default":{".",";","?","!"}| ending_punc,
                   "zh": {"。","？"}
                   }

    def __init__(self,sentence_list,meta,src="conllu"):
        self.sentences = {src:sentence_list}
        self.meta = meta

    def __repr__(self):
        return "\n".join(map(repr,self.sentences.get("conllu",self.sentences["tok"])))
    
    def get_sentences(self,src="tok"):
        return self.sentences[src]
    
    def baseline_split(self,lang="default"):
        """default split for languages where we have issues re-aligning tokens for various reasons
        
        this just splits at every token that is a hard punctuations 
        """
        sentence_id = 1
        sentences = []
        current = []
        orig_doc = self.sentences["tok"][0]
        for token in orig_doc:
            current.append(token)
            if token.lemma in self._hard_punct.get(lang,"default"):
                sentences.append(Sentence(current,meta))
                meta = {"doc_id":orig_doc.meta["doc_id"],
                    "sent_id" : sentence_id,
                    "text": " ".join([x.form for x in current])
                    }
                current = []
                sentence += 1
        if current!=[]:
            meta = {"doc_id":orig_doc.meta["doc_id"],
                    "sent_id" : sentence_id,
                    "text": " ".join([x.form for x in current])
                    }
            sentences.append(Sentence(current,meta))
        return sentences

    def ersatz_split(self,doc,lang='default-multilingual',candidates="en"):
        result = split(model=lang,
                       text=doc, output=None, 
                       batch_size=16, 
                       candidates=candidates,#'multilingual', 
                       cpu=True, columns=None, delimiter='\t') 
        return result
    
    def stanza_split(self,orig_doc,lang):
        nlp = stanza.Pipeline(lang=lang, processors='tokenize',download_method=DownloadMethod.REUSE_RESOURCES)
        doc = nlp(orig_doc)
        sentences = []
        for s in doc.sentences: 
            sentences.append(" ".join([t.text for t in s.tokens]))
        return sentences
        #for i, sentence in enumerate(doc.sentences): for token in sentence.tokens / token.text

    def trankit_split(self,orig_doc,lang,pipeline):
        trk_sentences = pipeline.ssplit(orig_doc)
        sentences = []
        for s in trk_sentences["sentences"]:
            sentences.append(s["text"])
        return sentences
    # TODO: debug option to for warnings on/off
    def _remap_tokens(self,split_sentences):
        """remap tokens from sentence splitting to the token original information"""
        #return split_sentences
        # if this fails, there's been a bug: count of tokens is different in original text, and total 
        # of split sentences
        # TODO: this is bound to happen, but the output should keep the original token count; how ?
        # TODO: REALIGN by detecting split tokens
        orig_token_nb = sum(map(len,self.sentences["tok"]))
        split_token_nb = len(list(chain(*[x.split() for x in split_sentences])))
        try:
            assert orig_token_nb==split_token_nb
        except:
            print("WARNING wrong nb of tokens",orig_token_nb,"initially but",split_token_nb,"after split",file=sys.stderr)
        #raise NotImplementedError
        new_sentences = []
        position = 0
        skip_first_token = False
        # will only work when splitting tok files, not resplitting conllu
        orig_doc = self.sentences["tok"][0]
        for i,s in enumerate(split_sentences):
            new_toks = s.split()
            if skip_first_token:# see below
                new_toks = new_toks[1:]
            toks = orig_doc.toks[position:position+len(new_toks)]
            meta = {"doc_id":orig_doc.meta["doc_id"],
                    "sent_id" : i+1,
                    "text": " ".join([x.form for x in toks])
                    }
            new_tok_position = position
            shift = 0 # advance thru new tokens in case of erroneous splits
            # actual nb of tokens to advance in the original document
            # new tokens might include split token by mistake (tricky)
            new_toks_length = len(new_toks)
            for j in range(len(toks)):
                toks[j].id = j+1
                new_j = j + shift
                try:
                    assert toks[j].form==new_toks[new_j]
                    # a split token has been detected meaning it had a punctuation sign in it and makes a "fake" sentence
                    # it will be recovered in current sentence so should be skipped in the next one
                    skip_first_token = False
                except:
                    # TODO: check next token can be recovered
                    # pb with chinese punctuation difference codes ?
                    #print(f"WARNING === Token mismatch: {j,toks[j].form,new_toks[new_j]} \n {toks} \n {new_toks}",file=sys.stderr)
                    # first case: within the same sentence (unlikely if a token was split by a punctuation)
                    if j!= len(toks)-1:
                        if len(toks[j].form)!=len(new_toks[new_j]): # if same length this is probably just an encoding problem (chinese cases) so just ignore it
                            #print(f"INFO: split token still within the sentence {j,toks[j].form,new_toks[new_j]} ... should not happen",file=sys.stderr)
                            if toks[j].form==new_toks[new_j]+new_toks[new_j+1]:
                                #print(f"INFO: split token correctly identified as {j,toks[j].form,new_toks[new_j]+new_toks[new_j+1]} ... advancing to next one",file=sys.stderr)
                                shift = shift + 1
                    # second case: the sentence ends here and next token is in the next split sentence, which necessarily exists (?)
                    else:
                        if i+1<len(split_sentences):
                            next_sentence = split_sentences[i+1]
                            next_token = split_sentences[i+1].split()[0]
                            skip_first_token = True
                            if toks[j].form==new_toks[new_j]+next_token: 
                                pass
                                #print(f"INFO: token can be recoverd: ",end="",file=sys.stderr)
                            else:
                                pass
                                #print(f"INFO: token can still not be recoverd: ",end="",file=sys.stderr)
                            #print(toks[j].form,new_toks[new_j]+next_token,file=sys.stderr)
                        else:
                            pass
                            #print(f"WARNING === unmatched token at end of document",new_toks[new_j],file=sys.stderr)
                            # in theory should not happen
                    # the next starting position has to be put back ? no
                    # position = position - 1
            if len(toks)>0: # joining the first token might have generated an empty sentence
                new_sentences.append(Sentence(toks,meta))
                position = position + len(new_toks) - shift
            else:
                skip_first_token = False
        return new_sentences


    def sentence_split(self,model="ersatz",lang="default-multilingual",**kwargs):
        """
        call the sentence splitter to the actual document read as one from a tok file. 
        kwargs might contain an open "pipeline" (eg. trankit pipeline) to pass on downstream for splitting sentences, so that it is not re-created for each paragraph
        """
        # if we split, the doc has been read as only one sentence 
        # we ignore multi-word-expression at reading time, but if this needs to be changed, it will impact this line:
        doc = [x.form for x in self.sentences["tok"][0]] # if not(x.is_MWE())]
        doc = " ".join(doc)
        if model=="ersatz":
            # empirically seems better: "en" for all alphabet-based language 
            # (candidates = candidate location for sentence splitting)
            # not to be confused with the language of the model
            candidates = "en" if lang not in {"zh","th"} else "multilingual"
            new_sentences = self.ersatz_split(doc,lang=lang,candidates=candidates)
        elif model=="stanza":
            new_sentences = self.stanza_split(doc,lang=lang)
        elif model=="trankit":# initiliazed pipeline is passed on here
            new_sentences = self.trankit_split(doc,lang=lang,**kwargs)
        elif model=="baseline":
            new_sentences = self.baseline_split(lang=lang)
            self.sentences["split"] = new_sentences
        else:
            raise NotImplementedError
        if model!="baseline":
            self.sentences["split"] = self._remap_tokens(new_sentences)
        return self.sentences["split"]
    

    def search_word(self,word):
        return [s for s in self.sentences.get("split",[]) if word in s]

    def format(self,mode="split"):
        """format the document as disrpt format
        mode=original (sentences) or split (split_sentences)
        """
        target = self.sentences[mode]
        
        output = "\n".join([s.format()+"\n" for s in target])
        meta = f"# doc_id = {self.meta}\n"
        return meta+output+"\n"


class Corpus:
    META_types = {"newdoc_id":"doc_id",
                  "newdoc id":"doc_id",
                  "doc_id":"doc_id",
                  "sent_id":"sent_id",
                  "newturn_id":"newturn_id",
                  "newutterance":"newutterance",
                  "newutterance_id":"newutterance_id",
                  "text":"text",
                  }



    def __init__(self,data=None):
        """input to constructor is a string
        """
        if data:
            self.docs = self._parse(data.split("\n"))
    
    @staticmethod
    def _meta_parse(data_line):
        """ parse comments as they contain meta information"""
        if not("=" in data_line):# not a meta line
            return "",""
        info, value = data_line[1:].strip().split("=",1)
        info = info.strip()
        if info in Corpus.META_types:
            meta_type = Corpus.META_types[info]
        else:# TODO should send a warning
            #print("WARNING: bad meta line",info, value,data_line,file=sys.stderr)
            meta_type, value = "",""
        return meta_type,value.strip()

    def search_doc(self,docid):
        return [x for x in self.docs if x.meta==docid]

    def _parse(self,data_lines,src="tok"):
        """parse disrpt segmentation/connective files"""
        curr_token_list = []
        sentences = []
        docs = []
        s_idx = 0
        doc_idx = 0
        meta = {}
        
        for data_line in data_lines:
            data_line = data_line.strip()
            if data_line:
                # comments always include some meta info of the form "metatype = value", minimally the document id
                if data_line.startswith("#"):
                    meta_type,value = Corpus._meta_parse(data_line)
                    # start of a new doc, save previous one if it exists
                    if meta_type=="doc_id":
                        if doc_idx>0:
                            docs.append(Document(sentences,meta["doc_id"],src=src))
                        sentences = []
                        curr_token_list = []
                        s_idx = 0
                        meta = {}
                        doc_idx += 1
                    if meta_type!="":
                        meta[meta_type] = value
                else:
                    token, label = self.parse_token(meta, data_line)
                    # if this is a MWE, just ignore it. MWE have ids combining original token ids, eg "30-31"
                    # TODO: refactor in parse_token + boolean flag if ok
                    if not("-" in token[0]):
                        curr_token_list.append(Token(*token,label))
            else:# end of sentence
                meta["text"] = " ".join((x.form for x in curr_token_list))
                s_idx += 1
                # some corpora dont have ids for sentences
                if "sent_id" not in meta: 
                    meta["sent_id"] = s_idx
                sentences.append(Sentence(curr_token_list,meta))
                curr_token_list = []
                meta = {"doc_id":meta["doc_id"]}
        if len(curr_token_list)>0 or len(sentences)>0:# final sentence for final document
            meta["text"] = " ".join((x.form for x in curr_token_list))
            sentences.append(Sentence(curr_token_list,meta))
            docs.append(Document(sentences,meta["doc_id"],src=src))
        return docs

    def parse_token(self, meta, data_line):
        *token, label = data_line.split("\t")
        if len(token)==8:
            print("ERROR: missing label ",meta,token,file=sys.stderr)
            token = token + [label]
            label = '_'
        # needed because of errors in source of some corpora (russian with BOM kept as token; also bad reading of some chars)
        # to prevent token counts/tokenization from failing, they are replaced with '_'
        # token[1] is the form of the token
        if token[1] == BOM: token[1]="_"
        #if token[1] == '200�000': 
        #    print("GOTCHA")
        token[1] = token[1].replace(REPL_CHAR,"_")
        label_set = set(label.split("|"))
        label = (label_set & set(self.LABELS))
        if label==set():
            label= "_"
        else:
            label = label.pop()
        return token,label

    def from_file(self,filepath):
        """ 
        reads a conllu or tok file
        connlu has sentences, tok does not

        option to pass on a string instead of file path, mostly for testing

        TODO: should be a static method
        """
        self.filepath = filepath
        basename = os.path.basename(filepath)
        src = basename.split(".")[-1] # tok or connlu or split
        #print("src = ",src)
        with open(filepath,"r",encoding="utf8") as f:
            data_lines = f.readlines()
        self.docs = self._parse(data_lines,src=src)

    def format(self,mode="split",file=sys.stdout):
        if type(file)==str:
            file = open(file,"w")
        for d in self.docs:
            print(d.format(mode=mode),file=file)

    def align(self,filepath):
        """load conllu for corresponding tok file"""
        pass
        
    def sentence_split(self,model="ersatz",lang="default-multilingual",**kwargs):
        """apply a sentence splitter to the document, assuming this was read from 
        a .tok file
            
        kwargs might contain an open "pipeline" (eg. trankit pipeline) to pass on downstream for splitting sentences, so that it is not re-created for each paragraph

        """
        for doc in tqdm(self.docs):
            doc.sentence_split(model=model,lang=lang,**kwargs)
        

    def eval_sentences(self,mode="split"):
        """eval sentence beginning as segment beginning
        TODO rename -> precision
        
        only .tok for now but could be used to eval re-split of connlu
        more complex for pdtb: need to align tok and connlu
        """
        tp = 0
        total_s = 0
        labels = []
        for doc in self.docs:
            for s in doc.get_sentences(mode):
                if len(s.toks)==0: 
                    print("WARNING empty sentence in ",s.meta,file=sys.stderr)
                    break
                tp += (s.toks[0].label=="BeginSeg=Yes")
                total_s += 1
                labels.extend([x.label for x in s])
        counts = Counter(labels)
        return tp, total_s, counts["BeginSeg=Yes"]

class SegmentCorpus(Corpus):
    LABELS = ["_","BeginSeg=Yes"]

class ConnectiveCorpus(Corpus):
    LABELS = ["_","Seg=B-Conn","Seg=I-Conn"]

class RelationCorpus(Corpus):

    def from_file(self,filepath):
        pass

# ersatz existing language-specific models
# for ersatz 1.0.0:
# ['en', 'ar', 'cs', 'de', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'iu', 'ja', 
# 'kk', 'km', 'lt', 'lv', 'pl', 'ps', 'ro', 'ru', 'ta', 'tr', 'zh', 'default-multilingual']
# missing disrpt languages/what candidates ? nl, pt, it -> en?  thai -> multilingual


if __name__=="__main__":
    # testing 
    import sys, os
    from pathlib import PurePath
    from ersatz import split, utils
    # ersatz existing language-specific models
    languages = utils.MODELS.keys()


    if len(sys.argv)>1:
        test_path = sys.argv[1]
    else:
        test_path = "../jiant/tests/test_data/eng.pdtb.pdtb/eng.pdtb.pdtb_debug.tok"
    test_path = "../jiant/tests/test_data/eng.pdtb.pdtb/eng.pdtb.pdtb_debug.tok"
    
    basename = os.path.basename(test_path)
    lang = basename.split(".")[0]
    lang = get_language(lang,"trankit")
    
    path = PurePath(test_path)
    #output_path = str(path.with_suffix(".split"))
    output_path = "out"
    
    if "pdtb" in test_path:
        corpus = ConnectiveCorpus()
    else:
        corpus = SegmentCorpus()
    corpus.from_file(test_path)
    
    #print(corpus.docs[0].sentences[11].display(segment=True))
    doc1 = corpus.docs[0]
    s0 = doc1.sentences["tok"][0]
    res = doc1.sentence_split()
    # check that number of token is conserved by sentence splitting
    #assert sum(map(len,doc1.sentences))==len(list(chain(*[x.split() for x in res])))
    pipeline = trankit.Pipeline(lang,gpu=True)
    corpus.sentence_split(model="trankit",lang=lang,pipeline=pipeline)
    tp, tot, all = corpus.eval_sentences()
    print(tp, tot, all)
    #print(corpus.docs[0].split_sentences[0].toks[0].format())
    corpus.format(file=output_path)