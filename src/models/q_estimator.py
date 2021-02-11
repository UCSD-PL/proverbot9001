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

import argparse

from typing import List, Tuple, Optional, Dict, Any
from abc import ABCMeta, abstractmethod
from pathlib_revised import Path2

from format import TacticContext



class QEstimator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, inputs: List[Tuple[TacticContext, str, float]]) -> List[float]:
        pass

    @abstractmethod
    def train(self, samples: List[Tuple[TacticContext, str, float, float]],
              batch_size: Optional[int] = None,
              num_epochs: int = 1,
              show_loss: bool = False) -> None:
        pass

    @abstractmethod
    def save_weights(self, filename: Path2, args: argparse.Namespace) -> None:
        pass
    @abstractmethod
    def load_saved_state(self, args: argparse.Namespace,
                         unparsed_args: List[str],
                         metadata: Any,
                         state: Dict[str, Any]) -> None:
        pass
