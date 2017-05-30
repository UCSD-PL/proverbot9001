Require Import Omega.

Lemma l : forall n: nat, (S n) > n.
Proof.
  induction n ; [try reflexivity | idtac ; try intro].
  2: omega.
  assert (1 = 1) ; auto.
Qed.

Lemma k : forall n: nat, (S n) > n.
Proof.
induction n.
  omega.
  auto.
  intros.
