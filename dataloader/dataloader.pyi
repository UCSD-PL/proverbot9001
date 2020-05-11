
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

class ScrapedTactic:
    relevant_lemmas : List[str]
    prev_tactics: List[str]
    prev_hyps: List[str]
    prev_goal: str
    tactic: str

class DataloaderArgs:
    max_distance: int
    max_string_distance: int
    max_length: int
    max_premises: int
    num_keywords: int
    num_relevance_samples: int
    keywords_file: str
    context_filter: str

@dataclass
class ProofContext:
    lemmas : List[str]
    tactics : List[str]
    hyps : List[str]
    goal : str

class TokenMap:
    ...

PickleableIndexer = Tuple[int, Dict[str, int], bool]
PickleableTokenizer = Tuple[bool, int, int, Dict[str, int]]
PickleableFeaturesTokenMap = Tuple[Dict[str, int],
                                   Dict[str, int],
                                   Dict[str, int]]

PickleableFPAMetadata = Tuple[PickleableIndexer,
                              PickleableTokenizer,
                              PickleableFeaturesTokenMap]

def scraped_tactics_from_file(filename : str, num_tactics : Optional[int]) -> List[ScrapedTactic]:
    ...

def features_polyarg_tensors(args : DataloaderArgs, filename : str) -> Tuple[PickleableFPAMetadata,
                                                                             Tuple[
                                                                                 List[List[List[int]]],
                                                                                 List[List[List[float]]],
                                                                                 List[int],
                                                                                 List[List[int]],
                                                                                 List[List[int]],
                                                                                 List[List[float]],
                                                                                 List[int],
                                                                                 List[int]],
                                                                             Tuple[List[int], int]]:
    ...

def features_polyarg_tensors_with_meta(args : DataloaderArgs, filename : str, meta : PickleableFPAMetadata) -> \
    Tuple[PickleableFPAMetadata,
          Tuple[
              List[List[List[int]]],
              List[List[List[float]]],
              List[int],
              List[List[int]],
              List[List[int]],
              List[List[float]],
              List[int],
              List[int]],
          Tuple[List[int], int]]:
    ...

def sample_fpa(args : DataloaderArgs, metadata : PickleableFPAMetadata,
               relevant_lemmas : List[str],
               prev_tactics : List[str],
               hypotheses : List[str],
               goal : str) -> \
               Tuple[
                   List[List[List[int]]],
                   List[List[List[float]]],
                   List[int],
                   List[List[int]],
                   List[List[int]],
                   List[List[float]]]:
    ...

def sample_fpa_batch(args : DataloaderArgs, metadata : PickleableFPAMetadata,
                     context_batch : List[ProofContext]) -> \
                     Tuple[
                         List[List[List[int]]],
                         List[List[List[float]]],
                         List[int],
                         List[List[int]],
                         List[List[int]],
                         List[List[float]]]:
    ...

def decode_fpa_result(args : DataloaderArgs, metadata : PickleableFPAMetadata,
                      hyps : List[str], goal: str, tac_idx: int, arg_idx: int) -> str:
    ...

def features_vocab_sizes(tmap: TokenMap) -> Tuple[List[int], int]:
    ...

def get_num_tokens(metadata : PickleableFPAMetadata) -> int:
    ...

def get_num_indices(metadata : PickleableFPAMetadata) -> int:
    ...

def get_word_feature_vocab_sizes(metadata : PickleableFPAMetadata) -> List[int]:
    ...

def get_vec_features_size(metadata: PickleableFPAMetadata) -> int:
    ...
