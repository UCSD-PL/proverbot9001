#!/usr/bin/env python3

from typing import Dict, List, Union, Tuple

class TacticPredictor:
    def getOptions(self) -> List[Tuple[str, str]]:
        assert False, "Need to override getOptions!"
        pass
    def __init__(self, options : Dict[str, Union[int, str]]) -> None:
        pass
    def predictKTactics(self, in_data : Dict[str, str], k : int) -> List[str]:
        assert False, "You can't predict on the base class!"
        pass
    def predictKTacticsWithLoss(self, in_data : Dict[str, str],
                                k : int, correct : str) -> \
        Tuple[List[str], float]:
        assert False, "You can't predict on the base class!"
        pass
