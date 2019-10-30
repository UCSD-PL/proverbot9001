
type unit0 =
| Tt

type bool =
| True
| False

val negb : bool -> bool

type nat =
| O
| S of nat

type 'a list =
| Nil
| Cons of 'a * 'a list

type comparison =
| Eq
| Lt
| Gt

val compOpp : comparison -> comparison

val add : nat -> nat -> nat

type positive =
| XI of positive
| XO of positive
| XH

type n =
| N0
| Npos of positive

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos :
 sig
  val succ : positive -> positive

  val add : positive -> positive -> positive

  val add_carry : positive -> positive -> positive

  val pred_double : positive -> positive

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val eqb : positive -> positive -> bool

  val coq_Nsucc_double : n -> n

  val coq_Ndouble : n -> n

  val coq_lxor : positive -> positive -> n
 end

module N :
 sig
  val succ_double : n -> n

  val double : n -> n

  val eqb : n -> n -> bool

  val div2 : n -> n

  val even : n -> bool

  val odd : n -> bool

  val coq_lxor : n -> n -> n
 end

module Z :
 sig
  val double : z -> z

  val succ_double : z -> z

  val pred_double : z -> z

  val pos_sub : positive -> positive -> z

  val add : z -> z -> z

  val opp : z -> z

  val sub : z -> z -> z

  val compare : z -> z -> comparison

  val leb : z -> z -> bool
 end

type ad = n

type 'a map =
| M0
| M1 of ad * 'a
| M2 of 'a map * 'a map

val mapGet : 'a1 map -> ad -> 'a1 option

val mapPut1 : ad -> 'a1 -> ad -> 'a1 -> positive -> 'a1 map

val mapPut : 'a1 map -> ad -> 'a1 -> 'a1 map

type fSet = unit0 map

val mapFold1 :
  'a2 -> ('a2 -> 'a2 -> 'a2) -> (ad -> 'a1 -> 'a2) -> (ad -> ad) -> 'a1 map
  -> 'a2

val mapFold :
  'a2 -> ('a2 -> 'a2 -> 'a2) -> (ad -> 'a1 -> 'a2) -> 'a1 map -> 'a2

val dmin : ('a1 -> 'a1 -> bool) -> 'a1 -> 'a1 -> 'a1

val ddmin : ('a1 -> 'a1 -> bool) -> 'a1 option -> 'a1 option -> 'a1 option

val ddle : ('a1 -> 'a1 -> bool) -> 'a1 option -> 'a1 option -> bool

val ddplus : ('a1 -> 'a1 -> 'a1) -> 'a1 option -> 'a1 -> 'a1 option

type 'd cGraph1 = 'd map map

val all_min :
  ('a1 -> 'a1 -> bool) -> (ad -> 'a1 -> 'a1 option) -> 'a1 map -> 'a1 option

val cG_add :
  ('a1 -> 'a1 -> bool) -> 'a1 cGraph1 -> ad -> ad -> 'a1 -> 'a1 map map

val ad_1_path_dist_1 :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1 cGraph1 -> ad ->
  ad -> fSet -> nat -> 'a1 option

type 'd cGForm =
| CGleq of ad * ad * 'd
| CGeq of ad * ad * 'd
| CGand of 'd cGForm * 'd cGForm
| CGor of 'd cGForm * 'd cGForm
| CGimp of 'd cGForm * 'd cGForm
| CGnot of 'd cGForm

type 'd cGSForm =
| CGSleq of ad * ad * 'd
| CGSand of 'd cGSForm * 'd cGSForm
| CGSor of 'd cGSForm * 'd cGSForm

val cG_test_ineq :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
  cGraph1 -> nat -> ad -> ad -> 'a1 -> bool

val cGS_solve_1 :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> bool
  -> 'a1 cGraph1 -> nat -> 'a1 cGSForm list -> nat -> bool

val fSize : 'a1 cGSForm -> nat

val cGS_solve :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> bool
  -> 'a1 cGSForm -> bool

val cGSeq : ('a1 -> 'a1) -> ad -> ad -> 'a1 -> 'a1 cGSForm

val cGSnot :
  ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> 'a1 -> 'a1 cGSForm -> 'a1 cGSForm

val cGFormSimplify :
  ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> 'a1 -> 'a1 cGForm -> 'a1 cGSForm

val cG_solve :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
  -> 'a1 cGForm -> bool

val cG_prove :
  'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
  -> 'a1 cGForm -> bool

type zCGForm =
| ZCGle of ad * ad
| ZCGge of ad * ad
| ZCGlt of ad * ad
| ZCGgt of ad * ad
| ZCGlep of ad * ad * z
| ZCGgep of ad * ad * z
| ZCGltp of ad * ad * z
| ZCGgtp of ad * ad * z
| ZCGlem of ad * ad * z
| ZCGgem of ad * ad * z
| ZCGltm of ad * ad * z
| ZCGgtm of ad * ad * z
| ZCGlepm of ad * ad * z * z
| ZCGgepm of ad * ad * z * z
| ZCGltpm of ad * ad * z * z
| ZCGgtpm of ad * ad * z * z
| ZCGeq of ad * ad
| ZCGeqp of ad * ad * z
| ZCGeqm of ad * ad * z
| ZCGeqpm of ad * ad * z * z
| ZCGzylem of ad * z
| ZCGzygem of ad * z
| ZCGzyltm of ad * z
| ZCGzygtm of ad * z
| ZCGzylepm of ad * z * z
| ZCGzygepm of ad * z * z
| ZCGzyltpm of ad * z * z
| ZCGzygtpm of ad * z * z
| ZCGzyeqm of ad * z
| ZCGzyeqpm of ad * z * z
| ZCGxzlep of ad * z
| ZCGxzgep of ad * z
| ZCGxzltp of ad * z
| ZCGxzgtp of ad * z
| ZCGxzlepm of ad * z * z
| ZCGxzgepm of ad * z * z
| ZCGxzltpm of ad * z * z
| ZCGxzgtpm of ad * z * z
| ZCGxzeqp of ad * z
| ZCGxzeqpm of ad * z * z
| ZCGzzlep of z * z
| ZCGzzltp of z * z
| ZCGzzgep of z * z
| ZCGzzgtp of z * z
| ZCGzzeq of z * z
| ZCGand of zCGForm * zCGForm
| ZCGor of zCGForm * zCGForm
| ZCGimp of zCGForm * zCGForm
| ZCGnot of zCGForm
| ZCGiff of zCGForm * zCGForm

val zCGtranslate : zCGForm -> z cGForm

val zCG_prove : zCGForm -> bool

val i2p : int -> positive

val i2a : int -> ad

val i2z : int -> z
