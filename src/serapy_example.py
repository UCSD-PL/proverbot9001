import coq_serapy
import multiprocessing
import random

def restart_coq(coq, coqargs, module, prelude, stmts):
    coq.kill()
    coq = None
    coq = coq_serapy.SerapiInstance(
        coqargs,
        module,
        prelude)
    for stmt in stmts:
        coq.run_stmt(stmt)
    return coq
    

multiprocessing.set_start_method('spawn')
stmts = [
    "From Coq Require Import List.",
    "From Coq Require Import Arith.",
    "From Coq Require Import Omega.",
    "From Coq Require Import Lia.",
    "Lemma rev_rev : forall (A : Type) (l : list A), rev (rev l) = l.",
    "Proof.",
    "intros A l.",
    "induction l.",
    "Admitted."
]

coqargs = ["sertop", "--implicit"]
module = "TestModule"
prelude = '/home/ubuntu/proverbot9001-errors/CompCert'
coq = coq_serapy.SerapiInstance(coqargs, module, prelude)

seen_stmts = []
for stmt in stmts:
    print(stmt) 
    try:
        if random.random() < 0.5:
            print("trying")
            print(id(coq))
            coq = restart_coq(coq, coqargs, module, prelude, seen_stmts)
            print(id(coq))
            print("------")
        coq.run_stmt(stmt)

    except coq_serapy.CoqAnomaly as e:
        print("caught")
        raise(e)
        coq = restart_coq(coq, coqargs, module, prelude, seen_stmts)
        coq.run_stmt(stmt)
    seen_stmts.append(stmt)

coq.kill()