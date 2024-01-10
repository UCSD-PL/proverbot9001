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

def search_thm_name(thm, coq):
    match = re.fullmatch("\w+ (?:SerTop\.)?([^\s]+)(?:\n.*)?", coq.locate_ident(thm))
    if match:
        return match.group(1)

class KNN_Lemma_Model:
    def __init__(self):
        # k-NN parameters
        # Would likely benefit from more descriptive names (these are taken from the coqhammer paper)
        self.T1 = 6   # Gives even more weight to higher scoring features
        self.T2 = 2.7 # Gives more weight to dependencies when estimating relevance of lemmas

        # Counter of how many different theorems each feature appears in
        self.feature_occurences = Counter()
            
        # Dict from theorem names to counters of features in that theorem
        self.theorem_feature_counts = {}

        # Dict from theorem names to the set of lemmas used in proving that theorem
        self.theorem_dependencies = {}

        # The lemma we're currently training on. Set by count_in_proof and unset by count_after_proof.
        self.cur_lemma_name = None

    def _theorems(self):
        return self.theorem_feature_counts.keys()

    def _tf_idf(self, feature, feature_counts):
        try:
            term_frequency = feature_counts[feature] / feature_counts.total()
            inverse_theorem_frequency = math.log(len(self.theorem_feature_counts)
                                      / self.feature_occurences[feature])
            return term_frequency * inverse_theorem_frequency
        except ZeroDivisionError:
            print(f"offending feature: {feature}")
            raise

    # These counting methods can double-count if called on the same proof twice.
    # Is it necessary TODO anything about this?

    def count_in_proof(self, coq):
        assert not self.cur_lemma_name, "We haven't finished counting " + self.cur_lemma_name + " !"

        self.cur_lemma_name = coq.module_prefix + coq.cur_lemma_name
        #print("In proof: " + self.cur_lemma_name)

        features = get_thm_features(coq)
        self.feature_counts = Counter(features[0]) + Counter(features[1])
        self.feature_occurences += Counter(list(self.feature_counts)) # Count each feature only once

        # We need to update these before we see the proof,
        # since we need feature weights for the current theorem
        # in order to pick out the lemmas we'll use to prove it!
        self.theorem_feature_counts[self.cur_lemma_name] = self.feature_counts
        # Dict from theorems to dicts from feature names to weights
        self.feature_weights = { thm: {f: self._tf_idf(f, feat_counts) for f in feat_counts}
                        for thm, feat_counts in self.theorem_feature_counts.items() }


    def count_after_proof(self, coq, cmds_run):
        assert self.cur_lemma_name, "We weren't in the middle of counting a theorem!"

        #print("After proof: " + self.cur_lemma_name)

        # FIXME: This disregards namespacing, so can break if trained on multiple modules.
        # We need to find a way to get the full, namespaced name BEFORE running the proof!
        #full_cur_lem_name = self.cur_lemma_name #_search_thm_name(self.cur_lemma_name, coq)
        #self.cur_lemma_name = None

        hypotheses = [hyp for cmd in cmds_run for hyp in
                    re.findall("(?:apply|rewrite)\s+(?:->|<-|with)?\s*([^\s;,)]*[^\s;,.)])", cmd)]

        lemmas_used = {name for name in [search_thm_name(hyp, coq) for hyp in hypotheses if hyp] if name is not None}

        self.theorem_dependencies[self.cur_lemma_name] = lemmas_used

        self.cur_lemma_name = None

    def train(self, prelude, *trainfiles):

        def _count_in_file(trainfile):
            with coq_serapy.SerapiContext(
                    ["sertop", "--implicit"],
                    None,
                    prelude) as coq:

                cmds_left = coq_serapy.load_commands(trainfile)

                while cmds_left:
                    cmds_left, _cmds_run = coq.run_into_next_proof(cmds_left)

                    self.count_in_proof(coq)

                    try:
                        cmds_left, cmds_run = coq.finish_proof(cmds_left)

                        self.count_after_proof(coq, cmds_run)

                    except AssertionError as e:
                        print(_cmds_run)
                        print(e)

        for trainfile in trainfiles:
            _count_in_file(trainfile)

        return self

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

    def get_relevant_lemmas(self, goal, num_lemmas):
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

    #total_model = KNN_Lemma_Model().train("/home/nmwaterman_umass_edu/proverbot9001/CompCert/", coqfile);
    model = KNN_Lemma_Model()

    with coq_serapy.SerapiContext(
            ["sertop", "--implicit"],
            None,
            "/home/nmwaterman_umass_edu/proverbot9001/CompCert/") as coq:

        cmds_left = coq_serapy.load_commands(coqfile)

        while cmds_left:
            cmds_left, _cmds_run = coq.run_into_next_proof(cmds_left)

            model.count_in_proof(coq)

            cur_lemma_name = model.cur_lemma_name

            predicted_lemmas = model.get_relevant_lemmas(cur_lemma_name, num_lemmas)
            total_lemmas_predicted += len(predicted_lemmas)

            try:
                cmds_left, cmds_run = coq.finish_proof(cmds_left)
                model.count_after_proof(coq, cmds_run)

                #full_lemma_name = search_thm_name(cur_lemma_name, coq)
                dependencies = model.theorem_dependencies[cur_lemma_name]
                #print(cur_lemma_name + " :")
                #print("Predicted: " + str(predicted_lemmas))
                #print("Actual: " + str(dependencies))
                #print()
                hits = [(i, lem) for i, lem in enumerate(predicted_lemmas) if lem in dependencies]
                correct_lemmas_predicted += len(hits)

                theorems_with_deps += 0 < len(dependencies)

                if hits:
                    theorems_hit += 1
                    if len(hits) == len(dependencies): theorems_fully_hit += 1

                #print(cur_lemma_name)
                #print(f"deps: {len(dependencies)}")
                #print(f"hits: {hits}\n")


            except AssertionError as e:
                #print(cmds_run)
                #print(e)
                pass

    print(f"\nWith {n} relevant lemmas...")
    print(f"Correctly predicted at least one lemma for {theorems_hit}/{theorems_with_deps} theorems requiring lemmas")
    print(f"Predicted all needed lemmas for {theorems_fully_hit}/{theorems_with_deps} theorems")
    print(f"{round(100*correct_lemmas_predicted/total_lemmas_predicted,1)}% of predicted lemmas were correct")

for n in range(1,20):
    test_lemma_prediction(n, "/home/nmwaterman_umass_edu/proverbot9001/CompCert/lib/Parmov.v")
