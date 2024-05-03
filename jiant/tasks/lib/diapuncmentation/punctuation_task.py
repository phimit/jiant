import re
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Union, Dict

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
        for token, label in zip(self.tokens, self.label_list):
            # Tokenize each "token" separately, assign label only to first token
            tokenized = tokenizer.tokenize(token)
            # If the token can't be tokenized, or is too long, replace with a single <unk>
            if len(tokenized) == 0 or len(tokenized) > ARBITRARY_OVERLY_LONG_WORD_CONSTRAINT:
                tokenized = [tokenizer.unk_token]
            # Mask 
            all_tokenized_tokens += tokenized
            padding_length = len(tokenized) - 1
            labels += [PunctuationTask.LABEL_TO_ID.get(label, None)] + [None] * padding_length
            label_mask += [1] + [0] * padding_length
        
        """ self.meta["tokenized"] = all_tokenized_tokens
        self.meta["labels"] = labels """

        return TokenizedExample(
            guid=self.guid,
            tokens=all_tokenized_tokens,
            labels=labels,
            label_mask=label_mask,
            meta=self.meta
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

        """ self.meta["input_ids"] = input_set.input_ids
        self.meta["input_mask"] = input_set.input_mask
        self.meta["label_ids"] = padded_labels
        self.meta["label_mask"] = padded_label_mask """

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            label_ids=np.array(padded_labels),
            label_mask=np.array(padded_label_mask),
            tokens=unpadded_inputs.unpadded_tokens,
            meta=self.meta
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

class PunctuationTask(Task):

    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.TAGGING
    LABELS = ["O", "B-.", "B-?", "B-,"]
    ORIG_LABELS = ["_", ".", "?", ","]
    FORMAT_LABELS = {cin:cin if cin!="_" else "" for cin in ORIG_LABELS}
    LABEL_MAP = dict(zip(ORIG_LABELS, LABELS))
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_labels(self):
        return len(self.LABELS)

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        curr_token_list, curr_label_list = [], []
        punc_signs = set('.,?!-"')
        # Read
        off_idx = 0
        for idx, line in enumerate(read_file_lines(path, "r", encoding="utf-8")):
            meta = ["","",""]
            line = line.strip().lower()
            # remove segmentatin marks, and unpredicted punctuation signs: " -
            line = re.sub('\||["-]', "", line)
            # consider ! as .
            line = line.replace("!",".")
            line_items = line.split(" ")
            for i, item in enumerate(line_items):
                if item in cls.ORIG_LABELS:
                    continue
                if i + 1 < len(line_items):
                    curr_token_list.append(item)
                    curr_label_list.append(cls.LABEL_MAP.get(line_items[i + 1], "O"))
                else:
                    curr_token_list.append(item)
                    curr_label_list.append("O")
            """ meta= {
                "orig_text": line,
                "base_tokens" : curr_token_list,
                "base_labels" : curr_label_list
            } """
            meta[2] = " ".join(curr_token_list)
            examples.append(
                Example(
                    guid=f"{set_type}-{idx}",
                    tokens=curr_token_list,
                    label_list= curr_label_list,
                    meta=meta
                )
            )
            curr_token_list, curr_label_list = [], []
        return examples


    @classmethod
    def format_predictions(cls, info_labels, data):
        """
        output predictions as saved in eg val_preds.p in a readable format

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
            tokens = sent_string.split()
            # this is were the label_mask should be used
            label_mask = info_labels[i]["label_mask"]
            relevant_preds = preds[i][label_mask]
            outlabels = [cls.FORMAT_LABELS[orig_labels[j]] for j in relevant_preds]
            try: # nb of predictions does not always match number of tokens if some instance is longer than max_seq_length
                # can be also an error reading metadata 
                assert len(outlabels)==len(tokens)
            except AssertionError:
                print(f"preds/tokens have different numbers, missing prediction set to default class",file=sys.stderr)
                print(f"docid={doc_id},sentence_id={sentence_id},sent={sent_string}",file=sys.stderr)
                print(f"out={len(outlabels)},tokens={len(tokens)}",file=sys.stderr)
                #raise AssertionError
                missing_preds = [cls.FORMAT_LABELS[orig_labels[default_label_idx]]]*(len(tokens)-len(outlabels))
                outlabels.extend(missing_preds)

            output.append(" ".join(["".join((tok,outlabels[n])) for (n,tok) in enumerate(tokens)]))

        return output
