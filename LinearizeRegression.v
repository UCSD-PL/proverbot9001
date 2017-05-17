From Coq Require Import Bool.

Theorem linerarizeMeEasy : forall a b, a = b \/ a && b = false.
Proof.
  intros; destruct a; intuition; destruct b; intuition.
Qed.

Theorem linerarizeMeHard : forall a b, a = true \/ b && a = false.
Proof.
  intros [|]; [ intros; left | intros ].
  reflexivity.
  right; destruct b; reflexivity.
Qed.
