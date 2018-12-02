import multiprocessing
import itertools
import io
from format import read_tuple

from context_filter import ContextFilter, get_context_filter

from typing import Tuple, List, Optional, Iterable

RawDataset = Iterable[Tuple[List[str], str, str]]

def read_text_data_worker__(lines : List[str]) -> RawDataset:
    def worker_generator():
        with io.StringIO("".join(lines)) as f:
            t = read_tuple(f)
            while t:
                yield t
                t = read_tuple(f)
    return list(worker_generator())

def read_text_data(data_path : str,  max_size:Optional[int]=None) -> RawDataset:
    with multiprocessing.Pool(None) as pool:
        line_chunks = file_chunks(data_path, 32768)
        data_chunks = pool.imap_unordered(read_text_data_worker__, line_chunks)
        result = list(itertools.islice(itertools.chain.from_iterable(data_chunks),
                                       max_size))
        return result
def get_text_data(data_path : str, context_filter_name : str,
                  max_tuples : Optional[int]=None, verbose : bool = False) -> RawDataset:
    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    _print("Reading dataset...")
    raw_data = read_text_data(data_path)
    filtered_data = list(itertools.islice(filter_data(raw_data, get_context_filter(context_filter_name)), max_tuples))
    _print("Got {} input-output pairs ".format(len(filtered_data)))
    return filtered_data

def filter_data(data : RawDataset, pair_filter : ContextFilter) -> RawDataset:
    return ((hyps, goal, tactic)
            for ((hyps, goal, tactic), (next_hyps, next_goal, next_tactic)) in
            zip(data, itertools.islice(data, 1, None))
            if pair_filter({"goal": goal, "hyps" : hyps}, tactic,
                           {"goal": next_goal, "hyps" : next_hyps}))

def file_chunks(filepath : str, chunk_size : int):
    with open(filepath, 'r') as f:
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if len(chunk) == chunk_size:
                while chunk[-1] != "-----\n":
                    nextline = f.readline()
                    if not nextline:
                        break
                    chunk += [nextline]
                    assert len(chunk) < chunk_size * 2
            elif len(chunk) == 0:
                return
            yield chunk
