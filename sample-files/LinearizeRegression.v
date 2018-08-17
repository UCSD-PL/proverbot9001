Require Import Bool.

Theorem linerarizeMeEasy : forall a b, a = b \/ a && b = false.
Proof.
  intros; destruct a; intuition; destruct b; intuition.
Qed.

Theorem linerarizeMeMedium : forall a b, a = true \/ b && a = false.
Proof.
  intros [|]; [ intros; left | intros ].
  reflexivity.
  right; destruct b; reflexivity.
Qed.

Ltac break_and :=
  match goal with
  | [ H : _ /\ _ |- _ ] => destruct H
  end.

Theorem linerarizeMeHard :
  forall (A B C D E F : Prop)
    (ABC: A -> B /\ C)
    (BD: B -> D)
    (CE: C -> E)
    (DEF: D /\ E -> F)
    (a : A), F.
Proof with (split; [ apply BD | apply CE ]; assumption).
  intros.
  destruct (ABC a).
  apply DEF...
Qed.
