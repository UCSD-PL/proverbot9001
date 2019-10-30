
type nat =
| O
| S of nat

type ('a, 'b) sum =
| Inl of 'a
| Inr of 'b

type ('a, 'b) prod =
| Pair of 'a * 'b

val snd : ('a1, 'a2) prod -> 'a2

type comparison =
| Eq
| Lt
| Gt

val compOpp : comparison -> comparison

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type ('a, 'p) sigT =
| ExistT of 'a * 'p

type sumbool =
| Left
| Right

val pred : nat -> nat

val add : nat -> nat -> nat

module Nat :
 sig
  val max : nat -> nat -> nat
 end

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos :
 sig
  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg
 end

module Coq_Pos :
 sig
  val succ : positive -> positive

  val add : positive -> positive -> positive

  val add_carry : positive -> positive -> positive

  val pred_double : positive -> positive

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  val succ_double_mask : mask -> mask

  val double_mask : mask -> mask

  val double_pred_mask : positive -> mask

  val sub_mask : positive -> positive -> mask

  val sub_mask_carry : positive -> positive -> mask

  val sub : positive -> positive -> positive

  val mul : positive -> positive -> positive

  val size_nat : positive -> nat

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val ggcdn :
    nat -> positive -> positive -> (positive, (positive, positive) prod) prod

  val ggcd :
    positive -> positive -> (positive, (positive, positive) prod) prod

  val of_succ_nat : nat -> positive
 end

module Z :
 sig
  val double : z -> z

  val succ_double : z -> z

  val pred_double : z -> z

  val pos_sub : positive -> positive -> z

  val add : z -> z -> z

  val opp : z -> z

  val mul : z -> z -> z

  val compare : z -> z -> comparison

  val sgn : z -> z

  val abs : z -> z

  val of_nat : nat -> z

  val to_pos : z -> positive

  val ggcd : z -> z -> (z, (z, z) prod) prod
 end

val z_lt_dec : z -> z -> sumbool

val z_lt_ge_dec : z -> z -> sumbool

val z_lt_le_dec : z -> z -> sumbool

val pow_pos : ('a1 -> 'a1 -> 'a1) -> 'a1 -> positive -> 'a1

type q = { qnum : z; qden : positive }

val qnum : q -> z

val qden : q -> positive

val inject_Z : z -> q

val qplus : q -> q -> q

val qmult : q -> q -> q

val qopp : q -> q

val qminus : q -> q -> q

val qinv : q -> q

val qlt_le_dec : q -> q -> sumbool

val qpower_positive : q -> positive -> q

val qpower : q -> z -> q

val qred : q -> q

val nat_log_inf : positive -> nat

val nat_log_sup : positive -> nat

type r = { cauchy : (nat -> q); modulus : (nat -> nat) }

val cauchy : r -> nat -> q

val modulus : r -> nat -> nat

val inject_Q : q -> r

type rpos = nat

val rplus : r -> r -> r

val ropp : r -> r

val rminus : r -> r -> r

type rlt = rpos

type rpos_alt = (nat, nat) sigT

val rpos_alt_1 : r -> rpos -> rpos_alt

val rpos_alt_2 : r -> rpos_alt -> rpos

val rcompare : r -> r -> rlt -> r -> (rlt, rlt) sum

val sqr2 : q -> q

val sqr2_h : q -> nat -> q

val sqr2_alpha : nat -> nat

val sqr2_w : nat -> nat

val sqr2_apply : r -> r

val sqr2_incr : r -> r -> rlt -> rlt

val qlt_Rlt : q -> q -> rlt

type itvl = { lft : q; rht : q }

val lft : itvl -> q

val sqrt2_approx : nat -> itvl

val sqrt2 : r
