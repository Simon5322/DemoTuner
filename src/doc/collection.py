'''
Created on Apr 3, 2021

@author: immanueltrummer
'''
from collections import Counter, defaultdict
from dataclasses import dataclass

import models
import nlp.nlp_util
from doc.util import get_parameters, get_values
from parameters.util import decompose_val
import enum
# import models.util
import pandas as pd
import parameters.util
import re
import torch
# import nlp.nlp_util
from dbms.generic_dbms import ConfigurableDBMS
# from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import Any


class HintType(enum.IntEnum):
    """ Represents the type of tuning hint. """
    DISK_RATIO = 0,
    RAM_RATIO = 1,
    CORES_RATIO = 2,
    ABSOLUTE = 3

    def __str__(self):
        """ Return string representation of value. """
        return [
            'Relative (disk)', 'Relative (RAM)',
            'Relative (Cores)', 'Absolute Value'][self]


@dataclass
class TuningHint():
    """ Represents a single tuning hint, assigning a parameter to a value. """
    doc_id: str
    param: Any
    value: Any
    recommendation: str
    passage: str
    float_val: float
    val_unit: str
    hint_type: HintType
    affect: Any

    def __init__(
            self, doc_id, passage, recommendation,
            param, value, hint_type, affect):
        """ Initializes tuning hint for given passage. 
        
        Args:
            doc_id: document from which hint was extracted
            passage: A text passage containing the hint.
            param: match object referencing parameter in passage.
            recommendation: text passage with recommended value.
            value: match object referencing value in passage.
            hint_type: type of tuning hint.
        """
        self.doc_id = doc_id
        self.passage = passage
        self.recommendation = recommendation
        self.param = param
        self.value = value
        self.float_val, self.val_unit = decompose_val(value.group())
        self.hint_type = hint_type
        self.affect = affect

class DocCollection():
    """ Represents a collection of documents with tuning hints. """
    qa_pipeline = pipeline(
        'question-answering',
        model='deepset/roberta-base-squad2',
        tokenizer='deepset/roberta-base-squad2',
        )
    zsc_pipeline = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
        tokenizer='facebook/bart-large-mnli',
        )

    def __init__(self, docs_path, size_threshold):
        """ Reads tuning passages from a file. 
        
        Reads passages containing tuning hints from a text. Tries
        to filter passages to interesting ones. If given, a DBMS
        is used to filter to passages containing parameter names.
        
        Args:
            docs_path: path to document with tuning hints.
            dbms: database management system (optional).
            size_threshold: start new passage after so many tokens.
            filter_params: whether to filter hints by their parameters.
            use_implicit: whether to consider implicit hints.
        """
        #self.dbms = dbms
        self.size_threshold = size_threshold

        self.docs = pd.read_csv(docs_path)
        self.docs.fillna('', inplace=True)
        self.nr_docs = self.docs['filenr'].max()
        self.nr_passages = []  # 记录每个doc中passage的个数
        self.passages_by_doc = []  # 一个doc占位数组一个单元，其中每个passage用/n区分

        # 将每个doc的多个passages和到一起，
        for doc_id in range(self.nr_docs):
            passages = self._doc_passages(doc_id + 1)
            self.passages_by_doc.append(passages)  # 一个doc占位数组一个单元，其中每个passage用/n区分
            self.nr_passages.append(len(passages))

        # Prepare caching of tuning hints
        self.doc_to_hints = {}
        # Calculate statistics
        self.asg_counts, self.param_counts = self._assignment_stats()
        # Sort hints by parameter
        self.param_to_hints = self._hints_by_param()
        # Output a summary of data read
        print(f'Initializing documents from file {docs_path} ...')
        print('Sample of tuning hints:')
        print(self.docs.sample())
        print(f'Nr. documents read: {self.nr_docs}')
        print(f'Nr. passages by doc: {self.nr_passages}')
        print(f'Nr. mentions per assignment: {self.asg_counts.most_common()}')
        print(f'Nr. documents per parameter: {self.param_counts.most_common()}')

    def _doc_passages(self, doc_id):
        """ Extract text snippets from given document. """
        snippets_idx = self.docs['filenr'] == doc_id
        snippets = self.docs.loc[snippets_idx, 'sentence']
        # Join snippets into larger passages
        passages = []
        passage = []
        p_length = 0
        for snippet in snippets:
            s_length = nlp.nlp_util.tokenize(snippet)['input_ids'].shape[1]
            p_length += s_length
            if p_length > self.size_threshold:
                # Start new passage
                passages.append('\n'.join(passage))
                passage = [snippet]
                p_length = 0
            else:
                # Append snippet to passage
                passage.append(snippet)
                p_length += s_length
        return passages



    def _preprocess_passage(self, passage):
        """ Pre-processes text of a passage for hint extraction.
        
        Args:
            passage: raw passage for pre-processing
        
        Returns:
            passage text after pre-processing
        """
        # print(f'Before pre-processing: {passage}')
        for val_sep_unit in re.finditer(
                f'(\d+)\s(kb|mb|gb)', passage, re.IGNORECASE):
            digits = val_sep_unit.group(1)
            unit = val_sep_unit.group(2)
            val_unit = f'{digits}{unit}'
            passage = passage.replace(val_sep_unit.group(), val_unit)
        # print(f'After pre-processing: {passage}')
        return passage

    def get_hints(self, doc_id):
        """ Returns candidate tuning hints extracted from given document. 
        
        Returns:
            List of candidate tuning hints.
        """
        if doc_id in self.doc_to_hints:
            return self.doc_to_hints[doc_id]
        else:
            print(f'Creating hints for document {doc_id}')
            hints = []
            passages = self.passages_by_doc[doc_id]
            if len(passages) == 0:
                return
            for passage in passages:
                exp_passage = passage

                exp_passage = self._preprocess_passage(exp_passage)
                params = re.finditer(parameters.util.param_reg, exp_passage)
                p_names = set([p.group() for p in params])
                for p_name in p_names:
                    if True:         #if self.dbms.is_param(p_name):
                        affect = self.get_passage_affectType(p_name, exp_passage)
                        answer, score = self._extract_value(p_name, exp_passage)
                        if score > 0.05:
                            values = re.finditer(
                                parameters.util.value_reg, answer)
                            for value in values:
                                hint_type = self._classify_hint(
                                    p_name, passage, value)
                                param = re.search(p_name, exp_passage)
                                hint = TuningHint(
                                    doc_id, exp_passage, answer,
                                    param, value, hint_type, affect)
                                hints.append(hint)
                                print(f'Adding hint {hint} with confidence {score}')
                        else:
                            print(
                                f'Excluding recommendation "{answer}" for ' \
                                f'parameter "{p_name}" due to low confidence ' \
                                f'({score})')
            self.doc_to_hints[doc_id] = hints
            data = pd.DataFrame(hints)
            data.to_csv("../../hints/hinttest.csv", index=False, header=False, mode="a")
            return hints
    def get_passage_affectType(self, config, exp_passage):
        label3 = [f"{config} affects other configs", f"{config} is affected by other condition", "not affected "]
        affected = ""
        affecting = ""
        condition = [affecting, affected]

        affectType = self.zsc_pipeline(exp_passage, label3)
        score = affectType["scores"]
        max_index = score.index(max(score))  # always 0
        max_label = affectType["labels"][max_index]
        q_result = self.qa_pipeline(question=f"{config} affect or be affected by what", context=exp_passage)
        q_answer = q_result["answer"]

        index = -1
        if max_label == label3[0]:
            index = 0
            affecting = q_answer
            condition[index] = affecting

        elif max_label == label3[1]:
            index = 1
            affected = q_answer
            condition[index] = affected
        else:
            print("not valid lable")

        return max_label, condition[index]
    def _assignment_stats(self):
        """ Generate statistics on candidate parameter assignments. """
        asg_counter = Counter()
        param_counter = Counter()
        for doc_id in range(self.nr_docs):
            doc_asgs = set()
            doc_params = set()
            hints = self.get_hints(doc_id)
            if hints is not None:
                for hint in hints:
                    asg = (hint.param.group(), hint.value.group())
                    doc_asgs.add(asg)
                    doc_params.add(asg[0])
                for asg in doc_asgs:
                    asg_counter.update([asg])
                for param in doc_params:
                    param_counter.update([param])
        return asg_counter, param_counter

    def _classify_hint(self, p_name, passage, value):
        """ Classifies hint depending on recommendation type.
        
        Args:
            p_name: name of parameter
            passage: text recommending values
            value: one specific recommended value
        """
        value_str = value.group()
        if '%' in value_str:
            resources = ['Disk', 'RAM', 'Cores']
            labels = [f'{p_name}: {value_str} ({r})' for r in resources]
            result = self.zsc_pipeline(passage, labels)
            winner_label = result['labels'][0]
            winner_idx = labels.index(winner_label)
            if winner_idx == 0:
                return HintType.DISK_RATIO
            elif winner_idx == 1:
                return HintType.RAM_RATIO
            elif winner_idx == 2:
                return HintType.CORES_RATIO
            else:
                raise ValueError(f'Unknown label "{winner_label}"')
        else:
            return HintType.ABSOLUTE

    def _extract_value(self, p_name, passage):
        """ Extracts recommended parameter value from passage.
        
        Args:
            p_name: name of parameter for which to extract values
            passage: extract recommendations from this passage
        
        Returns:
            tuple: recommendation, confidence
        """
        question = f'Which values are recommended for {p_name}?'
        qa_input = {'question': question, 'context': passage}
        qa_result = self.qa_pipeline(qa_input)
        answer = qa_result['answer']
        score = qa_result['score']
        return answer, score

    def _hints_by_param(self):
        """ Maps parameters to corresponding hints. """
        param_to_hints = defaultdict(lambda: [])
        for doc_id, doc_hints in self.doc_to_hints.items():
            for hint in doc_hints:
                param = hint.param.group()
                param_to_hints[param].append((doc_id, hint))
        return param_to_hints
