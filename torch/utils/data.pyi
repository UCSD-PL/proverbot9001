from torch import Tensor
from typing import Iterator, Tuple, List

class TensorDataset:
      def __init__(self, *data : Tensor) -> None: ...
      ...

class DataLoader:
      def __init__(self, dataset : TensorDataset,
                   batch_size : int = ...,
                   shuffle : bool = ...,
                   pin_memory : bool = ...,
                   num_workers : int = ...) -> None: ...
      def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]: ...
