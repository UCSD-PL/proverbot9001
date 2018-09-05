#!/usr/bin/env python3

import csv
import argparse
import re

from typing import Dict, Any, List, Tuple, TypeVar, Callable

import nmf
import numpy as np

class SimpleEmbedding:
    def __init__(self) -> None:
        self.tokens_to_indices = {} #type: Dict[str, int]
        self.indices_to_tokens = {} #type: Dict[int, str]
    def encode_token(self, token : str) -> int :
        if token in self.tokens_to_indices:
            return self.tokens_to_indices[token]
        else:
            new_idx = len(self.tokens_to_indices)
            self.tokens_to_indices[token] = new_idx
            self.indices_to_tokens[new_idx] = token
            return new_idx

    def decode_token(self, idx : int) -> str:
        return self.indices_to_tokens[idx]
    def num_tokens(self) -> int:
        return len(self.indices_to_tokens)
    def has_token(self, token : str) -> bool :
        return token in self.tokens_to_indices

ConfusionMatrix = List[List[int]]

def build_confusion_matrix(filenames : List[str], max_rows : float = float("Inf")) -> \
    Tuple[ConfusionMatrix, SimpleEmbedding]:
    matrix : ConfusionMatrix = []
    embedding = SimpleEmbedding()
    rows_processed = 0
    for filename in filenames:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if rows_processed >= max_rows:
                    break
                if len(row) == 1:
                    continue
                else:
                    rows_processed += 1
                    assert len(row) == 9
                    command, hyps, goal, \
                        first_pred, first_grade, \
                        second_pred, second_grade, \
                        third_pred, third_grade = row
                    encoded_command = embedding.encode_token(get_stem(command))
                    encoded_prediction = embedding.encode_token(get_stem(first_pred))
                    while encoded_command >= len(matrix):
                        matrix += [[]]
                    if encoded_prediction >= len(matrix[encoded_command]):
                        matrix[encoded_command] += ([0] *
                                                    (encoded_prediction -
                                                     len(matrix[encoded_command])
                                                     + 1))
                    matrix[encoded_command][encoded_prediction] += 1
    print("Built matrix from {} rows.".format(rows_processed))
    return matrix, embedding

def print_matrix(matrix : ConfusionMatrix):
    print("matrix is ")
    for row in matrix:
        print(row)

def print_confusion_info(matrix : ConfusionMatrix, embedding : SimpleEmbedding,
                         num_classes : int, print_num : int = 10) -> None:
    counts = {}
    total_counts = {}
    for i, row in enumerate(matrix):
        for j, item in enumerate(row):
            counts[(i, j)] = item
            canonical_coord = (min(i, j), max(i, j))
            if canonical_coord in total_counts:
                total_counts[canonical_coord] += item
            else:
                total_counts[canonical_coord] = item

    total_sorted_counts = sorted(list(counts.items()),
                                 key=lambda p : p[1],
                                 reverse=True)
    mistakes = [((i, j), count) for (i, j), count in total_sorted_counts if
                i != j]
    correct  = [((i, j), count) for (i, j), count in total_sorted_counts if
                i == j]
    print("Mistakes:")
    for ((i, j), count) in mistakes[:print_num]:
        print("{} times: {} and {} were confused ({} first for second, {} second for first)"
              .format(count, embedding.decode_token(i), embedding.decode_token(j),
                      counts[(i, j)] if (i, j) in counts else 0,
                      counts[(j, i)] if (j, i) in counts else 0))
    print("Correct Predictions:")
    for ((i, j), count) in correct[:print_num]:
        print("{} times: {}"
              .format(count, embedding.decode_token(i)))

    print("Clustering:")
    w, h = nmf.nmf(extend_conf_matrix(matrix), num_classes)
    partitions = multipartition(list(enumerate(w)), \
                                lambda i_cluster_factors: np.argmax(i_cluster_factors[1]))
    clusters = [[x[0] for x in partition]
                for partition in partitions]
    for i, cluster in enumerate(clusters):
        print("Cluster #{}:".format(i))
        for item in sorted(cluster, key=lambda i: sum(w[i]), reverse=True):
            print(embedding.decode_token(item))
        print()

    # print(nmf.nmf(extend_conf_matrix(matrix), num_classes))
    pass

T = TypeVar('T')
def multipartition(xs : List[T], f : Callable[[T], int]):
    result : List[T] = []
    for x in xs:
        assert x != None
        i = f(x)
        while i >= len(result):
            result += [[]]
        result[i] += [x]
    return result

def extend_conf_matrix(matrix : ConfusionMatrix):
    max_length = max([len(row) for row in matrix])
    return [row + [0] * (max_length - len(row)) for row in matrix]

def main() -> None:

    parser = argparse.ArgumentParser(description="Produce a confusiong matrix "
                                     "from proverbot9001 report csv files")
    parser.add_argument("filenames", nargs="+", help="csv file names")
    parser.add_argument("--num-classes", dest="num_classes", default=5, type=int)
    args = parser.parse_args()

    matrix, embedding = build_confusion_matrix(args.filenames)
    print_confusion_info(matrix, embedding, args.num_classes)

def kill_comments(string: str) -> str:
    result = ""
    depth = 0
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if depth == 0:
                result += string[i]
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i:i+2] == '(*':
                depth += 1
            if depth == 0:
                result += string[i]
            if string[i-1:i+1] == '*)':
                depth -= 1
            if string[i] == '"' and string[i-1] != '\\':
               in_quote = True
    return result

def get_stem(tactic : str) -> str:
    tactic = kill_comments(tactic).strip()
    if re.match("[-+*\{\}]", tactic):
        return tactic
    if re.match(".*;.*", tactic):
        return tactic
    match = re.match("^\(?(\w+).*", tactic)
    assert match, "tactic \"{}\" doesn't match!".format(tactic)
    return match.group(1)

if __name__ == "__main__":
    main()
