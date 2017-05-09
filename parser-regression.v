From Coq Require Import Bool.Bool.
From Coq Require Import ZArith.

(*
. .. .a a. .( (. ). .) {. .{ .} }.
(* . .. .a a. .( (. ). .) {. .{ .} }. *)
" *) "
. .. .a a. .( (. ). .) {. .{ .} }.
*)

Set Implicit Arguments.

Notation "[ x ; .. ; y ]" := (cons x .. (cons y nil) ..).
Notation "{ x ; .. ; y }" := (cons x .. (cons y nil) ..).

CoInductive stream A := { hd : A; tl : stream A }.

Definition expand {A} (s : stream A) := (Build_stream s.(hd) s.(tl)).

Theorem test : [1; 2] = {1; 2}.
Proof using.
  {-+ pose proof ( 1 + ( - 2 ) )%Z.
         * reflexivity...
  }
Qed.
