
val negb : bool -> bool

type nat =
| O
| S of nat

type 'a option =
| Some of 'a
| None

type ('a, 'b) prod =
| Pair of 'a * 'b

val fst : ('a1, 'a2) prod -> 'a1

val snd : ('a1, 'a2) prod -> 'a2

type comparison =
| Eq
| Lt
| Gt

val add : nat -> nat -> nat

module Nat :
 sig
  val leb : nat -> nat -> bool
 end

type positive =
| XI of positive
| XO of positive
| XH

type n =
| N0
| Npos of positive

module Pos :
 sig
  val succ : positive -> positive

  val compare_cont : comparison -> positive -> positive -> comparison

  val eqb : positive -> positive -> bool

  val coq_Nsucc_double : n -> n

  val coq_Ndouble : n -> n

  val coq_lxor : positive -> positive -> n

  val iter_op : ('a1 -> 'a1 -> 'a1) -> positive -> 'a1 -> 'a1

  val to_nat : positive -> nat
 end

module N :
 sig
  val eqb : n -> n -> bool

  val div2 : n -> n

  val even : n -> bool

  val odd : n -> bool

  val coq_lxor : n -> n -> n

  val to_nat : n -> nat
 end

type ad = n

type 'a map =
| M0
| M1 of ad * 'a
| M2 of 'a map * 'a map

val mapGet : 'a1 map -> ad -> 'a1 option

val newMap : 'a1 map

val mapPut1 : ad -> 'a1 -> ad -> 'a1 -> positive -> 'a1 map

val mapPut : 'a1 map -> ad -> 'a1 -> 'a1 map

val bDDzero : n

val bDDone : n

type bDDvar = ad

val bDDcompare : bDDvar -> bDDvar -> comparison

val ad_S : ad -> n

val max : nat -> nat -> nat

type bDDstate = (bDDvar, (ad, ad) prod) prod map

val initBDDstate : (bDDvar, (ad, ad) prod) prod map

type bDDsharing_map = ad map map map

val initBDDsharing_map : ad map map map

val bDDshare_lookup : bDDsharing_map -> bDDvar -> ad -> ad -> ad option

val bDDshare_put :
  bDDsharing_map -> bDDvar -> ad -> ad -> ad -> bDDsharing_map

type bDDconfig = (bDDstate, (bDDsharing_map, ad) prod) prod

val initBDDconfig :
  ((bDDvar, (ad, ad) prod) prod map, (ad map map map, n) prod) prod

val bDDalloc :
  bDDconfig -> bDDvar -> ad -> ad -> (((bDDvar, (ad, ad) prod) prod map,
  (bDDsharing_map, n) prod) prod, ad) prod

val bDDmake : bDDconfig -> bDDvar -> ad -> ad -> (bDDconfig, ad) prod

val var : bDDconfig -> ad -> bDDvar

val low : bDDconfig -> ad -> ad

val high : bDDconfig -> ad -> ad

type bDDneg_memo = ad map

val bDDneg_memo_lookup : bDDneg_memo -> ad -> ad option

val bDDneg_memo_put : bDDneg_memo -> ad -> ad -> ad map

type bDDor_memo = ad map map

val initBDDor_memo : ad map map

val bDDor_memo_put : bDDor_memo -> ad -> ad -> ad -> ad map map

val bDDor_memo_lookup : bDDor_memo -> ad -> ad -> ad option

val initBDDneg_memo : bDDneg_memo

val bDDneg_1_1 :
  bDDconfig -> bDDneg_memo -> ad -> nat -> ((bDDconfig, ad) prod,
  bDDneg_memo) prod

val bDDor_1_1 :
  bDDconfig -> bDDor_memo -> ad -> ad -> nat -> (bDDconfig, (ad, bDDor_memo)
  prod) prod

val bDDneg :
  bDDconfig -> bDDneg_memo -> ad -> (bDDconfig, (ad, bDDneg_memo) prod) prod

val bDDor :
  bDDconfig -> bDDor_memo -> ad -> ad -> (bDDconfig, (ad, bDDor_memo) prod)
  prod

val bDDand :
  bDDconfig -> bDDneg_memo -> bDDor_memo -> ad -> ad -> (bDDconfig, (ad,
  (bDDneg_memo, bDDor_memo) prod) prod) prod

val bDDimpl :
  bDDconfig -> bDDneg_memo -> bDDor_memo -> ad -> ad -> (bDDconfig, (ad,
  (bDDneg_memo, bDDor_memo) prod) prod) prod

val bDDiff :
  bDDconfig -> bDDneg_memo -> bDDor_memo -> ad -> ad -> (bDDconfig, (ad,
  (bDDneg_memo, bDDor_memo) prod) prod) prod

val bDDvar_make : bDDconfig -> bDDvar -> (bDDconfig, ad) prod

type bool_expr =
| Zero
| One
| Var of bDDvar
| Neg of bool_expr
| Or of bool_expr * bool_expr
| ANd of bool_expr * bool_expr
| Impl of bool_expr * bool_expr
| Iff of bool_expr * bool_expr

val bDDof_bool_expr :
  bDDconfig -> bDDneg_memo -> bDDor_memo -> bool_expr -> (bDDconfig, (ad,
  (bDDneg_memo, bDDor_memo) prod) prod) prod

val is_tauto : bool_expr -> bool
