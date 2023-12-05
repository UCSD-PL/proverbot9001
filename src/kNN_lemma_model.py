from lark import (Lark, Transformer, Tree, Token)
from itertools import chain
from collections import Counter
from dataclasses import dataclass
import argparse
import re
import math
import os

import coq_serapy

sexp_parser = Lark(r"""
        %import common (WS)
        %ignore WS

        ?start: exp

        ?exp: slist
            | ATOM
        
        slist: "(" exp (exp)* ")"

        ATOM: /[^()\s\t\n]+/
            | /\(\)/
    """, parser='lalr')

class ToFeatureTree(Transformer):
    def slist(self, items):
        def filter_items(items):
            for item in items:
                match item:
                    case Token("WS", _):
                        pass
                    case Token("ATOM", name):
                        yield name
                    case ["Var", ["Id", _]]:
                        yield "X"
                    case ["Rel", _]:
                        yield "X"
                    case [_, ["MPfile", ["DirPath", pathlist]], ["Id", name]]:
                        yield ".".join([*reversed([p[1] for p in pathlist]), name])
                    case ["Ind", [[name, *_], *_]]:
                        yield name
                    case ["Const", [name, *_], *_]:
                        yield name
                    case ["Construct", [[[prefix, _], name], *_]]:
                        yield f"{prefix}.{name}"
                    case ["App", func, args]:
                        # unpack case branches as if they were additional arguments
                        args = [*chain(*[arg[1:] \
                                if hasattr(arg, "__getitem__") and arg[0] == "case" \
                                else [arg] for arg in args])]
                        yield [func, args]
                    case ["Lambda", _, _, body] | ["Prod", _, _, body]:
                        yield body
                    case ["Case", *_, branches]:
                        yield ["case", branches]
                    case ["LetIn", *_, body]:
                        yield body
                    case _:
                        yield item
        return list(filter_items(items))

def extract_AST_from_goals_query(sexplist):
    match sexplist:
        case ["Answer", _, ["ObjList", [["CoqGoal", [["goals", [[_, ["ty", ast], *_]]], *_]]]]]:
            return ast

def extract_features(ast):
    match ast:
        case [parent, children]:
            child_features = (*map(extract_features, children),)
            adjacent_features = set().union(*map(lambda t: t[0], child_features))
            downstream_features = set().union(*map(lambda t: t[1], child_features))
            #print(f"\nparent: {parent}\ncf: {child_features}\nadj: {adjacent_features}\ndown: {downstream_features}")
            
            if parent == "case":
                return {}, {*downstream_features, *adjacent_features}
            else:
                return {parent, *{f"{parent}-{cf}" for cf in adjacent_features}}, {*downstream_features, *adjacent_features}
            #return {parent, *child_features, *{f"{parent}-{cf}" for cf in child_features}}
        case [_, _, *_]:
            assert False, f"Bad AST: {ast}"
        case leaf:
            return {leaf}, {}

def get_thm_features(coq):
    return extract_features(
            extract_AST_from_goals_query(
            ToFeatureTree().transform(
            sexp_parser.parse(
                coq.backend._ask_text('(Query () Goals)')
            ))))

def test_feature_extraction(tree):
    trantree = ToFeatureTree().transform(tree)
    ast = extract_AST_from_goals_query(trantree)
    print(f"\n\nast: {ast}")
    features = extract_features(ast)
    print(f"\nfeatures: {features}")

# "/home/nmwaterman_umass_edu/proverbot9001/CompCert/lib/Parmov.v"

class KNN_Lemma_Model:
    def __init__(self, *trainfiles):
        # k-NN parameters
        # Would likely benefit from more descriptive names (these are taken from the coqhammer paper)
        self.T1 = 6   # Gives even more weight to higher scoring features (per coqhammer)
        self.T2 = 2.7 # Gives more weight to dependencies when estimating relevance of lemmas

        # Counter of how many different theorems each feature appears in
        self.feature_occurences = Counter()
            
        # Dict from theorem names to counters of features in that theorem
        self.theorem_feature_counts = {}

        # Dict from theorem names to the set of lemmas used in proving that theorem
        self.theorem_dependencies = {}

        self.train(trainfiles)

    def _theorems(self):
        return self.theorem_feature_counts.keys()

    def train(self, trainfiles, stop_at=None):
    
        def _tf_idf(feature, feature_counts):
            try:
                term_frequency = feature_counts[feature] / feature_counts.total()
                inverse_theorem_frequency = math.log(len(self.theorem_feature_counts)
                                          / self.feature_occurences[feature])
                return term_frequency * inverse_theorem_frequency
            except ZeroDivisionError:
                print(f"offending feature: {feature}")
                raise

        def _count_in_file(trainfile):
            cmds_left = coq_serapy.load_commands(trainfile)

            def search_thm_name(thm, coq):
                match = re.fullmatch("\w+ ([^\s]+)(?:\n.*)?", coq.locate_ident(thm))
                if match: return match.group(1)

            with coq_serapy.SerapiContext(
                    ["sertop", "--implicit"],
                    None,
                    "/home/nmwaterman_umass_edu/proverbot9001/CompCert/") as coq: # TODO: Factor out coq context into object member
                while cmds_left:
                    cmds_left, _cmds_run = coq.run_into_next_proof(cmds_left)

                    cur_lemma_name = coq.cur_lemma_name

                    if cur_lemma_name == stop_at:
                        # Because this is the theorem we're currently trying to prove
                        # (or for some other reason)
                        # We can't look at its proof or any after it.
                        return False # Don't look at any more files

                    if cur_lemma_name in self._theorems():
                        # We've already seen this one; skip!
                        cmds_left, _cmds_run = coq.finish_proof(cmds_left)
                        continue

                    features = get_thm_features(coq)
                    feature_counts = Counter(features[0]) + Counter(features[1])

                    try:
                        cmds_left, cmds_run = coq.finish_proof(cmds_left)

                        full_cur_lem_name = search_thm_name(cur_lemma_name, coq)

                        self.theorem_feature_counts[full_cur_lem_name] = feature_counts;
                        self.feature_occurences += Counter(list(feature_counts)) # Count each feature only once

                        hypotheses = [hyp for cmd in cmds_run for hyp in 
                            re.findall("(?:apply|rewrite)\s+(?:->|<-|with)?\s*([^\s;,)]*[^\s;,.)])", cmd)]

                        #print(hypotheses)

                        lemmas_used = {name for name in [search_thm_name(hyp, coq) for hyp in hypotheses if hyp] if name is not None}
                        
                        self.theorem_dependencies[full_cur_lem_name] = lemmas_used

                    except AssertionError as e:
                        print(_cmds_run)
            return True # All good, look at the next file

        for trainfile in trainfiles:
            if not _count_in_file(trainfile):
                break # stop_at was in this file, so stop!
        
        # Dict from theorems to dicts from feature names to weights
        self.feature_weights = { thm: {f: _tf_idf(f, feat_counts) for f in feat_counts}
                for thm, feat_counts in self.theorem_feature_counts.items() }

    def _theorem_similarity(self, thm1, thm2):
        return sum(( (self.feature_weights[thm1][f] + self.feature_weights[thm2][f]) ** self.T1
                for f in self.feature_weights[thm1].keys() & self.feature_weights[thm2].keys() ))

    def _lemma_relevance(self, lem, goal, neighbors):
        return sum(( self._theorem_similarity(neighbor, goal) / len(self.theorem_dependencies[neighbor])
                for neighbor in neighbors if lem in self.theorem_dependencies[neighbor] )) * self.T2 \
            + (self._theorem_similarity(lem, goal) if lem in neighbors else 0)

    def _k_neighbors(self, goal, k):
        neighbors = sorted(self._theorems(), key=lambda thm: self._theorem_similarity(thm, goal), reverse=True)
        neighbors = [thm for thm in neighbors if thm != goal]
        return neighbors[:k]

    def get_relevant_lemmas(self, goal, num_lemmas, in_file=None):
        if in_file:
            self.train([in_file], stop_at=goal)

        relevant_lemmas = []
        for k in range(1,len(self._theorems())):
            neighbors = self._k_neighbors(goal, k)
            lems_and_rels = ((lem, self._lemma_relevance(lem, goal, neighbors)) for lem in self._theorems())
            relevant_lemmas = sorted({lem: relevance for lem, relevance in lems_and_rels if relevance > 0}, key=lambda pair: pair[1])
            if len(relevant_lemmas) >= num_lemmas: break
        return relevant_lemmas

def test_lemma_prediction(num_lemmas, coqfile):
    theorems_hit = 0
    theorems_fully_hit = 0
    theorems_with_deps = 0
    total_lemmas_predicted = 0
    correct_lemmas_predicted = 0

    theorems = KNN_Lemma_Model(coqfile).theorem_dependencies.items()

    model = KNN_Lemma_Model()

    for goal, deps in theorems:
        if deps:
            theorems_with_deps += 1

            relevant_lemmas = model.get_relevant_lemmas(goal, num_lemmas, in_file=coqfile)
            total_lemmas_predicted += len(relevant_lemmas)
            hits = [(i, lem) for i, lem in enumerate(relevant_lemmas) if lem in deps]
            correct_lemmas_predicted += len(hits)

            if hits:
                theorems_hit += 1
                if len(hits) == len(deps): theorems_fully_hit += 1

                #print(goal)
                #print(f"deps: {len(deps)}")
                #print(f"hits: {hits}\n")

    print(f"\nWith {n} relevant lemmas...")
    print(f"Correctly predicted at least one lemma for {theorems_hit}/{theorems_with_deps} theorems requiring lemmas")
    print(f"Predicted all needed lemmas for {theorems_fully_hit}/{theorems_with_deps} theorems")
    print(f"{round(100*correct_lemmas_predicted/total_lemmas_predicted,1)}% of predicted lemmas were correct")

for n in range(1,20):
    test_lemma_prediction(n, "/home/nmwaterman_umass_edu/proverbot9001/CompCert/lib/Parmov.v")
