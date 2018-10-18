#!/usr/bin/env python3

from typing import Dict, List, Union, Tuple, Iterable
from abc import ABCMeta, abstractmethod

class TacticPredictor(metaclass=ABCMeta):
    @abstractmethod
    def getOptions(self) -> List[Tuple[str, str]]: pass

    def __init__(self, options : Dict[str, Union[int, str]]) -> None:
        pass
    @abstractmethod
    def predictKTactics(self, in_data : Dict[str, Union[str, List[str]]], k : int) \
        -> Iterable[Tuple[str, float]]: pass
    @abstractmethod
    def predictKTacticsWithLoss(self, in_data : Dict[str, Union[str, List[str]]],
                                k : int, correct : str) -> \
        Tuple[List[Tuple[str, float]], float]: pass
