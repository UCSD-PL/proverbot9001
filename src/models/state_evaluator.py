##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

from format import TacticContext
from data import StateScore, StateEvaluationDataset, get_evaluation_data

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import (Generic, TypeVar, List, Iterable, Dict, Any)
from pathlib_revised import Path2
import argparse
import torch

class StateEvaluator(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    # Lower scores are better
    @abstractmethod
    def scoreState(self, state : TacticContext) -> float:
        pass

StateType = TypeVar('StateType')
class TrainableEvaluator(StateEvaluator, Generic[StateType],
                         metaclass=ABCMeta):
    def train(self, args : List[str]) -> None:
        argparser = argparse.ArgumentParser(self.description())
        self._add_args_to_parser(argparser)
        arg_values = argparser.parse_args(args)
        evaluation_data = get_evaluation_data(arg_values)
        save_states = self._optimize_model(evaluation_data, arg_values)
        for state in save_states:
            with open(arg_values.save_file, 'wb') as f:
                torch.save((self.shortname(), (arg_values, state)), f)

    @abstractmethod
    def description(self) -> str:
        pass
    @abstractmethod
    def shortname(self) -> str:
        pass
    @abstractmethod
    def _optimize_model(self, data : StateEvaluationDataset,
                       arg_values : argparse.Namespace) -> Iterable[StateType]:
        pass
    @abstractmethod
    def load_saved_state(self,
                         args : argparse.Namespace,
                         state : StateType) -> None:
        pass
    def _add_args_to_parser(self, parser : argparse.ArgumentParser,
                            default_values : Dict[str, Any] = {}) -> None:
        parser.add_argument("scrape_file", type=Path2)
        parser.add_argument("save_file", type=Path2)
        parser.add_argument("--num-threads", "-j", dest="num_threads", type=int,
                            default=default_values.get("num-threads", None))
        parser.add_argument("--max-tuples", dest="max_tuples", type=int,
                            default=default_values.get("max-tuples", None))
        parser.add_argument("--context-filter", dest="context_filter", type=str,
                            default=default_values.get("context-filter",
                                                       "default"))
        parser.add_argument('-v', '--verbose', action='count', default=0)
        parser.add_argument("--no-truncate_semicolons",
                            dest="truncate_semicolons",
                            action='store_false')
        parser.add_argument("--use-substitutions", dest="use_substitutions", type=bool,
                            default=default_values.get("use_substitutions", True))
        parser.add_argument("--print-keywords", dest="print_keywords",
                            default=False, action='store_const', const=True)
        parser.add_argument("--max-length", dest="max_length", type=int,
                            default=default_values.get("max-length", 30))

        pass
