import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Any, NamedTuple
import sys

# TODO: make meta information a namedtuple

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)


from jiant.tasks.lib.templates.shared import (
    labels_to_bimap,
    create_input_set_from_tokens_and_segments,
    construct_single_input_tokens_and_segment_ids,
    pad_single_with_feat_spec,
)
from jiant.utils.python.datastructures import zip_equal
from jiant.utils.python.io import read_file_lines



ARBITRARY_OVERLY_LONG_WORD_CONSTRAINT = 100
# In a rare number of cases, a single word (usually something like a mis-processed URL)
#  is overly long, and should not be treated as a real multi-subword-token word.
# In these cases, we simply replace it with an UNK token.


@dataclass
class Example(BaseExample):
    guid: str
    tokens: List[str]
    label_list: List[str]
    meta: List[str] = field(default_factory=list)

    def tokenize(self, tokenizer):
        all_tokenized_tokens = []
        labels = []
        label_mask = []
        for token, label in zip_equal(self.tokens, self.label_list):
            # Tokenize each "token" separately, assign label only to first token
            tokenized = tokenizer.tokenize(token)
            # If the token can't be tokenized, or is too long, replace with a single <unk>
            if len(tokenized) == 0 or len(tokenized) > ARBITRARY_OVERLY_LONG_WORD_CONSTRAINT:
                tokenized = [tokenizer.unk_token]
            all_tokenized_tokens += tokenized
            padding_length = len(tokenized) - 1
            labels += [DisrptTask.LABEL_TO_ID.get(label, None)] + [None] * padding_length
            label_mask += [1] + [0] * padding_length

        return TokenizedExample(
            guid=self.guid,
            tokens=all_tokenized_tokens,
            labels=labels,
            label_mask=label_mask,
            meta = self.meta
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List
    labels: List[Union[int, None]]
    label_mask: List[int]
    meta: List[str] = field(default_factory=list)

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.tokens,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        # Replicate padding / additional tokens for the label ids and mask
        if feat_spec.sep_token_extra:
            label_suffix = [None, None]
            mask_suffix = [0, 0]
            special_tokens_count = 3  # CLS, SEP-SEP
        else:
            label_suffix = [None]
            mask_suffix = [0]
            special_tokens_count = 2  # CLS, SEP
        unpadded_labels = (
            [None] + self.labels[: feat_spec.max_seq_length - special_tokens_count] + label_suffix
        )
        unpadded_labels = [i if i is not None else -1 for i in unpadded_labels]
        unpadded_label_mask = (
            [0] + self.label_mask[: feat_spec.max_seq_length - special_tokens_count] + mask_suffix
        )

        padded_labels = pad_single_with_feat_spec(
            ls=unpadded_labels,
            feat_spec=feat_spec,
            pad_idx=-1,
        )
        padded_label_mask = pad_single_with_feat_spec(
            ls=unpadded_label_mask,
            feat_spec=feat_spec,
            pad_idx=0,
        )

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            label_ids=np.array(padded_labels),
            label_mask=np.array(padded_label_mask),
            tokens=unpadded_inputs.unpadded_tokens,
            meta = self.meta
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_ids: np.ndarray
    label_mask: np.ndarray
    tokens: list
    meta: List[str] = field(default_factory=list)

@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_ids: torch.LongTensor
    label_mask: torch.LongTensor
    tokens: list
    #met: List[str] = field(default_factory=list)



class DisrptTask(Task):

    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.TAGGING
    # normalized during data reading to match sequence tagging assumption (BIO or sth like it)
    LABELS = ["O","B-Seg"]
    # original corresponding labels
    ORIG_LABELS = ["_","BeginSeg=Yes"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    RNN_MODULES = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }   
    META_DOC = {"newdoc id":0,"sent_id":1,"text":2}

    # PM TODO: 
    # - we reproduce panx subtask system here by giving a corpus name 
    # this influence the reading part too
    # for now, we set the default to gum to avoid problem downstream
    # - To modify the head (just a projection, but we can also have an LSTM): add parameters in config, 
    # declare them here. When the head is created, it has access to the task -> create the model from the task config
    # it's the easiest way to propagate task config to the task head
    # - can it be done for backbone too ? unlikely but must be checked 
    def __init__(self, name, path_dict, corpus="eng.rst.gum",format="conllu",**kwargs):
        """default constructor argument are path_dict to train/val/test and name of task
        
        additional for segmentation (may be in kwargs): 
           - corpus origin if we want to automate training every one of them at once (not implemented); explicit argument but must be in kwargs of config
           - format: conllu/plain: (not used) if we want to load plain format corpus, we'll need to 
             add a presegmenter when loading to avoid long sequence to crash transformers
             other option is to just generate a separate pre-segmented corpus (eg with ersatz)
           - recurrent_layer = None/lstm/gru
           - if previous arg is not none, will use an argument "rnn_config", which contains at least a "hidden_size" argument, and 
             may have also all RNN constructor optional argument (eg bidirectional, cf torch documentation)
        """
        super().__init__(name=name, path_dict=path_dict)
        self.corpus = corpus
        ## here we should overrides TAGGING init to add a RNN if we want
        ## (not so simple ... check how TAGGING is called at TASK creation + what goes to the HEAD -> how to override the head ?)
        recurrent_layer = kwargs.get("recurrent_layer")
        if recurrent_layer != "None": 
            self.rnn_type = recurrent_layer
            self.rnn_cfg = kwargs["rnn_config"]                          
            #self.rnn = recurrent_modules[recurrent_layer](input_size=hidden_size,**kwargs["rnn_config"])
            #output_size = kwargs["rnn_config"]["hidden_size"]
            #self.projection = nn.Linear(output_size, self.num_labels)
            self.classif_type = "recurrent_layer"
        else:
            self.classif_type = "simple"
        print(self.__dict__)



    @property
    def num_labels(self):
        return len(self.LABELS)

    def get_train_examples(self):
        return self._create_examples(data_path=self.path_dict["train"], set_type="train")

    def get_val_examples(self):
        return self._create_examples(data_path=self.path_dict["val"], set_type="val")

    def get_test_examples(self):
        return self._create_examples(data_path=self.path_dict["test"], set_type="test")

    @classmethod
    def _create_examples(cls, data_path, set_type):
        """disrpt21 format has 10 fields: 
            1 = token number in sentence or text
            2 = token
            3-9 = empty "_" or syntax info (lemma/supertag/postag/... etc)
            10 has segmentation label + info for syntax/deep syntax all separated by "|"

            set_type = train/test/dev if it makes any difference
        """
        curr_token_list, curr_label_list = [], []
        data_lines = read_file_lines(data_path, "r", encoding="utf-8")
        examples = []
        # TODO: namedtuple
        # store meta information: doc id, sentence id, sentence string (but each token sep by spaces)
        meta = ["","",""]
        idx = 0
        
        for data_line in data_lines:
            data_line = data_line.strip()
            if data_line:
                if data_line.startswith("#"):
                    info, value = data_line[1:].strip().split("=",1)
                    info = info.strip()
                    if info in cls.META_DOC:
                        meta[cls.META_DOC[info]] = value.strip()
                else:
                    _, token, *useless, labels = data_line.split("\t")
                    label_set = set(labels.split("|"))
                    # 0 = _, 1 = segment boudary
                    if cls.ORIG_LABELS[1] in label_set:
                        label = cls.LABELS[1]
                    else:
                        label = cls.LABELS[0]
                    
                    curr_token_list.append(token)
                    curr_label_list.append(label)
            else:
                # FIXED: metadata[sentence_string] should ALWAYS be stored from the list of tokens
                #if meta[2]=="":# some corpora dont put the list of tokens in commentary
                meta[2] = " ".join(curr_token_list)
                examples.append(
                    Example(
                        guid="%s-%s" % (set_type, idx),
                        tokens=curr_token_list,
                        label_list=curr_label_list,
                        meta = meta,
                    )
                )
                idx += 1
                curr_token_list, curr_label_list = [], []
                meta = [meta[0],"",""]
        if curr_token_list:
            #if meta[2]=="":# some corpora dont put the list of tokens in commentary
            meta[2] = " ".join(curr_token_list)
            examples.append(
                Example(guid="%s-%s" % (idx, idx), tokens=curr_token_list, label_list=curr_label_list,meta=meta)
            )
        return examples

    @classmethod
    def format_predictions(cls, info_labels, data):
        """
        output predictions as saved in eg val_preds.p in the original disrpt format

        info_labels: contains reference label mask corresponding to sub-tokenized input, saved in the cache
        necessary because the saved prediction tensor does not store explicitely the subtokenization 


        data is the content of the torch-saved predictions.
        it contains the keys: 
           "meta": is the list of instance meta information, here it should be a tuple as defined above: 
            (doc_id,sentence_id,sentence string)
            sentence string should be "tokenized": splitting on spaces will yield the list of tokens
           "preds": the vector of predictions on subtokens
        """
        default_label_idx = 0
        orig_labels = cls.ORIG_LABELS
        meta = data["meta"]
        preds = data["preds"]

        output = []

        for i,instance in enumerate(meta):
            doc_id, sentence_id, sent_string = instance
            if doc_id!="": 
                output.append(f"# doc_id: {doc_id}")
            output.append(f"# sentence_id: {sentence_id}")
            output.append(f"# text = {sent_string}")
            tokens = sent_string.split()
            # this is were the label_mask should be used
            label_mask = info_labels[i]["label_mask"]
            relevant_preds = preds[i][label_mask]
            outlabels = [orig_labels[j] for j in relevant_preds]
            try: # nb of predictions does not always match number of tokens if some instance is longer than max_seq_length
                # can be also an error reading metadata 
                assert len(outlabels)==len(tokens)
            except AssertionError:
                print(f"preds/tokens have different numbers, missing prediction set to default class",file=sys.stderr)
                print(f"docid={doc_id},sentence_id={sentence_id},sent={sent_string}",file=sys.stderr)
                print(f"out={len(outlabels)},tokens={len(tokens)}",file=sys.stderr)
                #raise AssertionError
                missing_preds = [orig_labels[default_label_idx]]*(len(tokens)-len(outlabels))
                outlabels.extend(missing_preds)
            for n,tok in enumerate(tokens):
                # disrpt format (no feature)
                # token_id token _ _ _ _ _ _ _ label
                output.append(f"{n+1}\t{tok}\t"+"\t".join(["_"]*7)+f"\t{outlabels[n]}")
            output.append("")
        return output
