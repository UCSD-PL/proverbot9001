#!/usr/bin/env python3

from typing import Dict, List, Union, Tuple, Iterable, NamedTuple
from abc import ABCMeta, abstractmethod

class Prediction(NamedTuple):
    prediction : str
    certainty : float

ContextInfo = Dict[str, Union[str, List[str]]]

class TacticPredictor(metaclass=ABCMeta):
    @abstractmethod
    def getOptions(self) -> List[Tuple[str, str]]: pass

    def __init__(self, options : Dict[str, Union[int, str]]) -> None:
        pass
    @abstractmethod
    def predictKTactics(self, in_data : ContextInfo, k : int) \
        -> List[Prediction]: pass
    @abstractmethod
    def predictKTacticsWithLoss(self, in_data : ContextInfo, k : int, correct : str) -> \
        Tuple[List[Prediction], float]: pass
    @abstractmethod
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[ContextInfo],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]: pass
