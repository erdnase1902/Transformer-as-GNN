from __future__ import absolute_import, division, print_function

import numpy as np
from allennlp.predictors.predictor import Predictor
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json
import os
from tqdm import tqdm
from model import BertForSequenceClassificationCustom
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
#sentence = "james ate some cheese whiasdflst thinking about the play ."
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

adj_method = "random" # untouched, random
random_ratio = 0.5

def cache_all_results(sentences, name, two_sentence=False):
    if os.path.exists(name):
        print('Loading from existing file {}'.format(name))
        with open(name) as f:
            return json.load(f)
    all_results = []
    if not two_sentence:
        for sentence in tqdm(sentences):
            tree = predictor.predict(sentence=sentence)
            all_results.append((tree["words"], tree["predicted_heads"]))
    else:
        for sentence_a, sentence_b in tqdm(sentences):
            tree_a = predictor.predict(sentence=sentence_a)
            tree_b = predictor.predict(sentence=sentence_b)
            all_results.append((tree_a["words"], tree_a["predicted_heads"], tree_b["words"], tree_b["predicted_heads"]))
    with open(name, "w") as f:
        json.dump(all_results, f)
    return all_results

global_statistics_edges = 0.0
global_statistics_all_edges = 0.0
def adj_matrix_parse(words, predicted_heads, do_parse = True):
    #tree = predictor.predict(sentence=sentence)
    #print(tree["words"], tree["predicted_heads"], tree["predicted_dependencies"])
    tokenized_words = []
    n = 0
    map_old_to_new = dict()

    # [0, 1, 4, 5, 2, 5, 6, 9, 7, 1] -> [-1, 0, 3, 4, 1, 4, 5, 8, 6, 0]
    for oldidx, word in enumerate(words):
        sub_tokens = []
        for token in tokenizer.basic_tokenizer.tokenize(word):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)     
        #sub_tokens = tokenizer.tokenize(word)
        map_old_to_new[oldidx] = list(range(n, n+len(sub_tokens))) # list(range(9, 10))
        n += len(sub_tokens)
        tokenized_words.append(sub_tokens)

    if not do_parse:
      return tokenized_words, None

    adj_mat = np.zeros(shape=(n,n))

    if adj_method == "untouched":
      return tokenized_words, np.ones(shape=(n,n))

    if adj_method == "random":
      for oldidx, word in enumerate(words):
        adj_mat[np.array(map_old_to_new[oldidx])[:, None], np.array(map_old_to_new[oldidx])] = 1
    
      num_conn_to_add = int( (random_ratio * n * n - adj_mat.sum(-1).sum(-1).item()) / 2)
      counter = 0
      for _ in range(10000):
        word1 = np.random.randint(0, n)
        word2 = np.random.randint(0, n)
        if word1 == word2 or adj_mat[word1, word2] == 1:
          continue
        adj_mat[word1, word2] = 1
        adj_mat[word2, word1] = 1
        counter += 1
        if counter == num_conn_to_add:
            break
      
      return tokenized_words, adj_mat

    for oldidx, word in enumerate(words):
      adj_mat[np.array(map_old_to_new[oldidx])[:, None], np.array(map_old_to_new[oldidx])] = 1

    for childidx, head in enumerate(predicted_heads):
      parent_idx = head - 1
      if not parent_idx==-1: 
        adj_mat[np.array(map_old_to_new[childidx])[:, None], np.array(map_old_to_new[parent_idx])] = 1
        adj_mat[np.array(map_old_to_new[parent_idx])[:, None], np.array(map_old_to_new[childidx])] = 1
      else:
        continue
    
    new_edges_to_add = int((random_ratio * n * n - adj_mat.sum(-1).sum(-1).item()) / 2)
    counter = 0
    if new_edges_to_add > 0:
        for _ in range(10000):
            word1 = np.random.randint(0, n)
            word2 = np.random.randint(0, n)
            if word1 == word2 or adj_mat[word1, word2] == 1:
                continue
            adj_mat[word1, word2] = 1
            adj_mat[word2, word1] = 1
            counter += 1
            if counter == new_edges_to_add:
                break
      
    #print(adj_mat)
    #adj_mat = np.ones(shape=(n,n))
    global global_statistics_edges
    global global_statistics_all_edges
    global_statistics_edges += adj_mat.sum(-1).sum(-1)
    global_statistics_all_edges += n * n
    return tokenized_words, adj_mat



import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, adj_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.adj_matrix = adj_matrix


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

from tqdm import tqdm
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, save_string):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    
    ########## get the parsed_results

    if examples[0].text_b:
        two_sentence = True
        sentences = [(example.text_a, example.text_b) for example in examples]
    else:
        two_sentence = False
        sentences = [example.text_a for example in examples]
    all_results = cache_all_results(sentences, name = "{}_{}.json".format(args.task_name, save_string), two_sentence=two_sentence)

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a, adj_matrix_a = adj_matrix_parse(all_results[ex_index][0], all_results[ex_index][1]) #tokenizer.tokenize(example.text_a)
        new_tokens_a = []
        for tmp in tokens_a:
          new_tokens_a.extend(tmp)
        tokens_a = new_tokens_a

        tokens_b = None
        if example.text_b:
            #tokens_b = tokenizer.tokenize(example.text_b)
            tokens_b, adj_matrix_b = adj_matrix_parse(all_results[ex_index][2], all_results[ex_index][3])
            new_tokens_b = []
            for tmp in tokens_b:
              new_tokens_b.extend(tmp)
            tokens_b = new_tokens_b

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            adj_matrix = torch.zeros(max_seq_length, max_seq_length)
            adj_matrix[0] = 1
            adj_matrix[:, 0] = 1
            adj_matrix_a = torch.from_numpy(adj_matrix_a)
            adj_matrix_b = torch.from_numpy(adj_matrix_b)
            adj_matrix[1:adj_matrix_a.size(0) + 1, 1:adj_matrix_a.size(0) + 1] = adj_matrix_a
            adj_matrix[len(adj_matrix_a)] = 1
            adj_matrix[:, len(adj_matrix_a)] = 1
            adj_matrix[adj_matrix_a.size(0) + 1 : adj_matrix_a.size(0) + 1 + adj_matrix_b.size(0), adj_matrix_a.size(0) + 1 : adj_matrix_a.size(0) + 1 + adj_matrix_b.size(0)] = adj_matrix_b
            adj_matrix[adj_matrix_a.size(0) + 1 + adj_matrix_b.size(0)] = 1
            adj_matrix[:, adj_matrix_a.size(0) + 1 + adj_matrix_b.size(0)] = 1
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            adj_matrix = torch.zeros(max_seq_length, max_seq_length)
            adj_matrix[0] = 1
            adj_matrix[:, 0] = 1
            adj_matrix_a = torch.from_numpy(adj_matrix_a)
            adj_matrix[1:adj_matrix_a.size(0) + 1, 1:adj_matrix_a.size(0) + 1] = adj_matrix_a
            adj_matrix[len(adj_matrix_a) + 1] = 1
            adj_matrix[:, len(adj_matrix_a) + 1] = 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              adj_matrix = adj_matrix.long()))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def get_adj_matrix(batch, method):
    # NOTE todo
    # method: untouched, random, syntax
    # 10 sentences, 20 words

    # I live in LA . [PAD]
    # I study CS at UCLA .

    # input_ids: 10 x 20
    # adj_matrx: 10 x 20 x 20. adj_matrix[0, 3, 4] the forth and fifth word in the first sentence
    # input_mask: 10 x 20 > 10 x 20 x 20
    
    input_ids, input_mask, segment_ids, label_ids = batch
    num_sentences = input_mask.shape[0]
    num_words = input_mask.shape[1]
    if method=='untouched':
      adj = np.ones(shape=(num_sentences, num_words, num_words))
      for idx in range(len(adj)):
        # np.fill_diagonal(adj[idx], 0)
        # input_mask[idx] is somthing like False, True, False, False, True, False, False, True, False, False, False, True, False, False, True, False, False, True, False, False
        # input_mask[2] tells us which words in the 3rd sentence are padded words
        # input_mask[2][5]==False tells us that in the 3rd sentence, the 6th word is a pad
        # adj[idx] is a 20x20 matrix
        # adj[idx][input_mask[idx]] is selecting rows 1, 4, 7, 11, 14, 17
        adj[idx][~input_mask[idx]] = 0
        adj[idx][:,~input_mask[idx]] = 0
        # fully connected graph
      return adj
    elif method=='random':
      """
      adj[idx] is the adj mat for sentence idx: 20x20 matrix
      """
      """
      1. Randomize the number of connections
      2. Randomize words to be connected
      """
      adj = np.ones(shape=(num_sentences, num_words, num_words))
      for idx in range(len(adj)):
        #np.fill_diagonal(adj[idx], 0)
        adj[idx][~input_mask[idx]] = 0
        adj[idx][:,~input_mask[idx]] = 0

        max_num_of_conn = int(np.sum(adj[idx])/2)
        num_conn_to_remove = np.random.randint(0, max_num_of_conn)
        for _ in range(num_conn_to_remove):
          word1 = np.random.randint(0, num_words)
          word2 = np.random.randint(0, num_words)
          if word1 == word2:
            continue
          adj[idx][word1, word2] = 0
          adj[idx][word2, word1] = 0
      return adj

      '''
      # psuedocode for randomly turn off connections
      rand_iteration = random.get(1~20);
      for in in range(rand_iteration) :
          word1 = rand.get(1~20)
          word2 = rand.get(1~20)
          assert(word1 != word2)
          remove connection(word1, word2)
      '''
      pass
    elif method=='syntax':
      # wait for Harold's clarafication
      raise NotImplementedError()
      pass
    else:
      raise NotImplementedError()
    

    pass

from attrdict import AttrDict
args = AttrDict()
args.data_dir = "glue_data/SST-2"
args.bert_model = "bert-base-uncased"
args.task_name = "SST-2"
args.output_dir = "output_{}_{}".format(adj_method, random_ratio)
args.cache_dir = "./"
args.max_seq_length = 128

args.do_train = True
args.do_eval = True
args.do_lower_case = True

args.train_batch_size = 32
args.eval_batch_size = 32
args.learning_rate = 2e-5
args.num_train_epochs = 3
args.warmup_proportion = 0.1
args.no_cuda = False
args.local_rank = -1
args.seed = 42
args.gradient_accumulation_steps =1
args.fp16 = False
args.loss_scale = 0
args.server_ip = ''
args.server_port = ''
        
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                        args.gradient_accumulation_steps))

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

#if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
#    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

task_name = args.task_name.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
output_mode = output_modes[task_name]

label_list = processor.get_labels()
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
# Prepare model
cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
model = BertForSequenceClassificationCustom.from_pretrained(args.bert_model,
            cache_dir=cache_dir,
            num_labels=num_labels)
if args.fp16:
    model.half()
model.to(device)
if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                            t_total=num_train_optimization_steps)

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)


global_step = 0
nb_tr_steps = 0
tr_loss = 0
if args.do_train:
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, "train")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    
    all_adj_matrix = torch.stack([f.adj_matrix for f in train_features], dim = 0).long()

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_adj_matrix)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        all_loss = []
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)

            # 10 sentences, 20 words

            # I live in LA . [PAD]
            # I study CS at UCLA .

            # input_ids: 10 x 20
            # adj_matrx: 10 x 20 x 20. adj_matrix[0, 3, 4] the forth and fifth word in the first sentence
            # input_mask: 10 x 20 > 10 x 20 x 20
            
            input_ids, input_mask, segment_ids, label_ids, adj_matrix = batch

            #adj_matrix = get_adj_matrix(batch, method = "syntax") # tensor: batch x sequence_length x sequence_length
            #adj_matrix = adj_matrix.to(device)

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None, custom_adj_matrix = adj_matrix)

            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            all_loss.append(loss.item())
            if step % 100 == 0 and step != 0:
                print('Step: {} loss: {}'.format(step, sum(all_loss[-100:]) / 100))
                
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassificationCustom.from_pretrained(args.output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
else:
    model = BertForSequenceClassificationCustom.from_pretrained(args.bert_model, num_labels=num_labels)
model.to(device)

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, "dev")
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_adj_matrix = torch.stack([f.adj_matrix for f in eval_features], dim = 0).long()
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_adj_matrix)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids, adj_matrix in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        adj_matrix = adj_matrix.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, custom_adj_matrix = adj_matrix
            )

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, all_label_ids.numpy())
    loss = tr_loss/nb_tr_steps if args.do_train else None

    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    result['loss'] = loss

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    # hack for MNLI-MM
    if task_name == "mnli":
        task_name = "mnli-mm"
        processor = processors[task_name]()

        if os.path.exists(args.output_dir + '-MM') and os.listdir(args.output_dir + '-MM') and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir + '-MM'):
            os.makedirs(args.output_dir + '-MM')

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
        
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

print(adj_method)
print(random_ratio)
print(global_statistics_edges/global_statistics_all_edges)