from collections import Counter
from typing import List, Dict, Tuple, Optional
import typing
import sys
import tqdm

# The algorithm implemented here is based on the
# https://huggingface.co/transformers/tokenizer_summary.html#byte-pair-encoding-bpe
def get_bpe_vocab(word_counts: Dict[str, int], merges: int) -> List[str]:
    print("Building vocab:")
    sys.stdout.flush()
    # Our base vocab is every character in any word in our counts.
    base_vocab = list(set([x for l in [list(word) for word in word_counts.keys()]
                           for x in l]))

    # Initialize our vocab to our base vocab
    vocab = list(base_vocab)
    # Build a list, for each word, of the current chunks that make it up and
    # its count. Because our initial vocab is base characters, this means
    # initially splitting each word into it's characters. eg: ("best", 14) ->
    # (["b", "e", "s", "t"], 14).
    word_breakdowns = [(list(word), count) for word, count in word_counts.items()]
    for i in tqdm.trange(merges):
        # Keep the count of each subsequent pair of chunks in our current
        # iteration.
        pair_counts: typing.Counter[Tuple[str, str]] = Counter()
        for chunks, count in word_breakdowns:
            for pair in zip(chunks[:-1], chunks[1:]):
                pair_counts[pair] += count
        # Get the most common chunk pairing
        most_common_pair, mcp_count = pair_counts.most_common(1)[0]
        # Update the word breakdowns to merge that pair
        new_word_breakdowns = []
        for chunks, count in word_breakdowns:
            new_chunks = []
            i = 0
            # Loop through the chunks for each word, and merge any subsequent
            # ones that match the most common pair.
            while i + 1 < len(chunks):
                if chunks[i] == most_common_pair[0] and \
                   chunks[i+1] == most_common_pair[1]:
                    new_chunks.append(most_common_pair[0] + most_common_pair[1])
                    i += 2
                else:
                    new_chunks.append(chunks[i])
                    if i + 2 == len(chunks):
                        new_chunks.append(chunks[i+1])
                    i += 1
            new_word_breakdowns.append((new_chunks, count))
        word_breakdowns = new_word_breakdowns
        # Finally, add the pair to the vocab list.
        vocab.append(most_common_pair[0] + most_common_pair[1])
    return vocab

