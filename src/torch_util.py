
from typing import overload, Callable, List, Tuple

import torch
import torch.cuda
import torch.autograd as autograd

from util import silent

def str_1d_long_tensor(tensor : torch.LongTensor):
    if (type(tensor) == autograd.Variable):
        tensor = tensor.data
    tensor = tensor.view(-1)
    return str(list(tensor))

def str_1d_float_tensor(tensor : torch.FloatTensor):
    if (type(tensor) == autograd.Variable):
        tensor = tensor.data
    tensor = tensor.view(-1)
    output = io.StringIO()
    print("[", end="", file=output)
    if tensor.size()[0] > 0:
        print("{:.4f}".format(tensor[0]), end="", file=output)
    for f in tensor[1:]:
        print(", {:.4f}".format(f), end="", file=output)
    print("]", end="", file=output)
    result = output.getvalue()
    output.close()
    return result

@overload
def _inflate(tensor : torch.LongTensor, times : int) -> torch.LongTensor: ...
@overload
def _inflate(tensor : torch.FloatTensor, times : int) -> torch.FloatTensor: ...

def _inflate(tensor : torch.Tensor, times : int) -> torch.Tensor:
    tensor_dim = len(tensor.size())
    if tensor_dim == 3:
        b = tensor.size(1)
        return tensor.repeat(1, 1, times).view(tensor.size(0), b * times, -1)
    elif tensor_dim == 2:
        return tensor.repeat(1, times)
    elif tensor_dim == 1:
        b = tensor.size(0)
        return tensor.repeat(times).view(b, -1)
    else:
        raise ValueError("Tensor can be of 1D, 2D, or 3D only. "
                         "This one is {}D.".format(tensor_dim))

def topk_with_filter(t : torch.FloatTensor, k : int, f : Callable[[float, int], bool]) \
    -> Tuple[torch.FloatTensor, torch.LongTensor]:
    all_certainties, all_idxs = t.topk(t.size()[0])
    certainties = []
    idxs = []
    for certainty, idx in zip(all_certainties, all_idxs):
        if f(certainty.item(), idx.item()):
            certainties.append(certainty)
            idxs.append(idx)
            if len(certainties) == k:
                break
    return FloatTensor(certainties), LongTensor(idxs)

with silent():
    use_cuda = torch.cuda.is_available()
    cuda_device = "cuda:0"

    # we want to use this in modules,
    # but we also want to compile those modules.

    # pytorch is extremely unhappy with references to
    # global variables.

    # therefore, we will bake the above static result
    # into these functions right now.

    if use_cuda:
        def maybe_cuda(component):
            return component.to(device=torch.device("cuda:0"))

        def LongTensor(x : List[int]):
            return torch.tensor(x,dtype=torch.long).to(device=torch.device("cuda:0"))

        def FloatTensor(x : List[float]):
            return torch.tensor(x,dtype=torch.float32).to(device=torch.device("cuda:0"))

        def ByteTensor(x : List[int]):
            return torch.tensor(x,dtype=torch.uint8).to(device=torch.device("cuda:0"))
    else:
        def maybe_cuda(component):
                return component

        def LongTensor(x : List[int]):
            return torch.tensor(x,dtype=torch.long)

        def FloatTensor(x : List[float]):
            return torch.tensor(x,dtype=torch.float32)

        def ByteTensor(x : List[int]):
            return torch.tensor(x,dtype=torch.uint8)
    # now these can be easily compiled into torchscript.

