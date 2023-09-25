from typing import (List, Optional, Dict, Tuple, Union, Any, Set,
                            Sequence, TypeVar, Callable, OrderedDict)
import coq2vec

class CachedObligationEncoder(coq2vec.CoqContextVectorizer):
    obl_cache: OrderedDict[Obligation, torch.FloatTensor]
    max_size: int
    def __init__(self, term_encoder: 'coq2vec.CoqTermRNNVectorizer',
            max_num_hypotheses: int, max_size: int=5000) -> None:
        super().__init__(term_encoder, max_num_hypotheses)
        self.obl_cache = OrderedDict()
        self.max_size = max_size #TODO: Add in arguments if desired
    def obligations_to_vectors_cached(self, obls: List[Obligation]) \
            -> torch.FloatTensor:
        encoded_obl_size = self.term_encoder.hidden_size * (self.max_num_hypotheses + 1)

        cached_results = []
        for obl in obls:
            r = self.obl_cache.get(obl, None)
            if r is not None:
                self.obl_cache.move_to_end(obl)
            cached_results.append(r)

        encoded = run_network_with_cache(
            lambda x: self.obligations_to_vectors(x).view(len(x), encoded_obl_size),
            [coq2vec.Obligation(list(obl.hypotheses), obl.goal) for obl in obls],
            cached_results)

        # for row, obl in zip(encoded, obls):
        #     assert obl not in self.obl_cache or (self.obl_cache[obl] == row).all(), \
        #         (self.obl_cache[obl] == row)

        for row, obl in zip(encoded, obls):
            self.obl_cache[obl] = row
            self.obl_cache.move_to_end(obl)
            if len(self.obl_cache) > self.max_size:
                self.obl_cache.popitem(last=False)
        return encoded
