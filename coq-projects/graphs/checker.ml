
type unit0 =
| Tt

type bool =
| True
| False

(** val negb : bool -> bool **)

let negb = function
| True -> False
| False -> True

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

(** val compOpp : comparison -> comparison **)

let compOpp = function
| Eq -> Eq
| Lt -> Gt
| Gt -> Lt

(** val add : nat -> nat -> nat **)

let rec add n0 m =
  match n0 with
  | O -> m
  | S p -> S (add p m)

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

module Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q -> XO (add_carry p q)
       | XO q -> XI (add p q)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q -> XI (add p q)
       | XO q -> XO (add p q)
       | XH -> XI p)
    | XH -> (match y with
             | XI q -> XO (succ q)
             | XO q -> XI q
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q -> XI (add_carry p q)
       | XO q -> XO (add_carry p q)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q -> XO (add_carry p q)
       | XO q -> XI (add p q)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q -> XI (succ q)
       | XO q -> XO (succ q)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | XI p ->
      (match y with
       | XI q -> compare_cont r p q
       | XO q -> compare_cont Gt p q
       | XH -> Gt)
    | XO p ->
      (match y with
       | XI q -> compare_cont Lt p q
       | XO q -> compare_cont r p q
       | XH -> Gt)
    | XH -> (match y with
             | XH -> r
             | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare =
    compare_cont Eq

  (** val eqb : positive -> positive -> bool **)

  let rec eqb p q =
    match p with
    | XI p0 -> (match q with
                | XI q0 -> eqb p0 q0
                | _ -> False)
    | XO p0 -> (match q with
                | XO q0 -> eqb p0 q0
                | _ -> False)
    | XH -> (match q with
             | XH -> True
             | _ -> False)

  (** val coq_Nsucc_double : n -> n **)

  let coq_Nsucc_double = function
  | N0 -> Npos XH
  | Npos p -> Npos (XI p)

  (** val coq_Ndouble : n -> n **)

  let coq_Ndouble = function
  | N0 -> N0
  | Npos p -> Npos (XO p)

  (** val coq_lxor : positive -> positive -> n **)

  let rec coq_lxor p q =
    match p with
    | XI p0 ->
      (match q with
       | XI q0 -> coq_Ndouble (coq_lxor p0 q0)
       | XO q0 -> coq_Nsucc_double (coq_lxor p0 q0)
       | XH -> Npos (XO p0))
    | XO p0 ->
      (match q with
       | XI q0 -> coq_Nsucc_double (coq_lxor p0 q0)
       | XO q0 -> coq_Ndouble (coq_lxor p0 q0)
       | XH -> Npos (XI p0))
    | XH ->
      (match q with
       | XI q0 -> Npos (XO q0)
       | XO q0 -> Npos (XI q0)
       | XH -> N0)
 end

module N =
 struct
  (** val succ_double : n -> n **)

  let succ_double = function
  | N0 -> Npos XH
  | Npos p -> Npos (XI p)

  (** val double : n -> n **)

  let double = function
  | N0 -> N0
  | Npos p -> Npos (XO p)

  (** val eqb : n -> n -> bool **)

  let rec eqb n0 m =
    match n0 with
    | N0 -> (match m with
             | N0 -> True
             | Npos _ -> False)
    | Npos p -> (match m with
                 | N0 -> False
                 | Npos q -> Pos.eqb p q)

  (** val div2 : n -> n **)

  let div2 = function
  | N0 -> N0
  | Npos p0 -> (match p0 with
                | XI p -> Npos p
                | XO p -> Npos p
                | XH -> N0)

  (** val even : n -> bool **)

  let even = function
  | N0 -> True
  | Npos p -> (match p with
               | XO _ -> True
               | _ -> False)

  (** val odd : n -> bool **)

  let odd n0 =
    negb (even n0)

  (** val coq_lxor : n -> n -> n **)

  let coq_lxor n0 m =
    match n0 with
    | N0 -> m
    | Npos p -> (match m with
                 | N0 -> n0
                 | Npos q -> Pos.coq_lxor p q)
 end

module Z =
 struct
  (** val double : z -> z **)

  let double = function
  | Z0 -> Z0
  | Zpos p -> Zpos (XO p)
  | Zneg p -> Zneg (XO p)

  (** val succ_double : z -> z **)

  let succ_double = function
  | Z0 -> Zpos XH
  | Zpos p -> Zpos (XI p)
  | Zneg p -> Zneg (Pos.pred_double p)

  (** val pred_double : z -> z **)

  let pred_double = function
  | Z0 -> Zneg XH
  | Zpos p -> Zpos (Pos.pred_double p)
  | Zneg p -> Zneg (XI p)

  (** val pos_sub : positive -> positive -> z **)

  let rec pos_sub x y =
    match x with
    | XI p ->
      (match y with
       | XI q -> double (pos_sub p q)
       | XO q -> succ_double (pos_sub p q)
       | XH -> Zpos (XO p))
    | XO p ->
      (match y with
       | XI q -> pred_double (pos_sub p q)
       | XO q -> double (pos_sub p q)
       | XH -> Zpos (Pos.pred_double p))
    | XH ->
      (match y with
       | XI q -> Zneg (XO q)
       | XO q -> Zneg (Pos.pred_double q)
       | XH -> Z0)

  (** val add : z -> z -> z **)

  let add x y =
    match x with
    | Z0 -> y
    | Zpos x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> Zpos (Pos.add x' y')
       | Zneg y' -> pos_sub x' y')
    | Zneg x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> pos_sub y' x'
       | Zneg y' -> Zneg (Pos.add x' y'))

  (** val opp : z -> z **)

  let opp = function
  | Z0 -> Z0
  | Zpos x0 -> Zneg x0
  | Zneg x0 -> Zpos x0

  (** val sub : z -> z -> z **)

  let sub m n0 =
    add m (opp n0)

  (** val compare : z -> z -> comparison **)

  let compare x y =
    match x with
    | Z0 -> (match y with
             | Z0 -> Eq
             | Zpos _ -> Lt
             | Zneg _ -> Gt)
    | Zpos x' -> (match y with
                  | Zpos y' -> Pos.compare x' y'
                  | _ -> Gt)
    | Zneg x' ->
      (match y with
       | Zneg y' -> compOpp (Pos.compare x' y')
       | _ -> Lt)

  (** val leb : z -> z -> bool **)

  let leb x y =
    match compare x y with
    | Gt -> False
    | _ -> True
 end

type ad = n

type 'a map =
| M0
| M1 of ad * 'a
| M2 of 'a map * 'a map

(** val mapGet : 'a1 map -> ad -> 'a1 option **)

let rec mapGet m a =
  match m with
  | M0 -> None
  | M1 (x, y) -> (match N.eqb x a with
                  | True -> Some y
                  | False -> None)
  | M2 (m1, m2) ->
    (match a with
     | N0 -> mapGet m1 N0
     | Npos p0 ->
       (match p0 with
        | XI p -> mapGet m2 (Npos p)
        | XO p -> mapGet m1 (Npos p)
        | XH -> mapGet m2 N0))

(** val mapPut1 : ad -> 'a1 -> ad -> 'a1 -> positive -> 'a1 map **)

let rec mapPut1 a y a' y' = function
| XO p' ->
  let m = mapPut1 (N.div2 a) y (N.div2 a') y' p' in
  (match N.odd a with
   | True -> M2 (M0, m)
   | False -> M2 (m, M0))
| _ ->
  (match N.odd a with
   | True -> M2 ((M1 ((N.div2 a'), y')), (M1 ((N.div2 a), y)))
   | False -> M2 ((M1 ((N.div2 a), y)), (M1 ((N.div2 a'), y'))))

(** val mapPut : 'a1 map -> ad -> 'a1 -> 'a1 map **)

let rec mapPut m x x0 =
  match m with
  | M0 -> M1 (x, x0)
  | M1 (a, y) ->
    (match N.coq_lxor a x with
     | N0 -> M1 (x, x0)
     | Npos p -> mapPut1 a y x x0 p)
  | M2 (m1, m2) ->
    (match x with
     | N0 -> M2 ((mapPut m1 N0 x0), m2)
     | Npos p0 ->
       (match p0 with
        | XI p -> M2 (m1, (mapPut m2 (Npos p) x0))
        | XO p -> M2 ((mapPut m1 (Npos p) x0), m2)
        | XH -> M2 (m1, (mapPut m2 N0 x0))))

type fSet = unit0 map

(** val mapFold1 :
    'a2 -> ('a2 -> 'a2 -> 'a2) -> (ad -> 'a1 -> 'a2) -> (ad -> ad) -> 'a1 map
    -> 'a2 **)

let rec mapFold1 neutral op f pf = function
| M0 -> neutral
| M1 (a, y) -> f (pf a) y
| M2 (m1, m2) ->
  op (mapFold1 neutral op f (fun a0 -> pf (N.double a0)) m1)
    (mapFold1 neutral op f (fun a0 -> pf (N.succ_double a0)) m2)

(** val mapFold :
    'a2 -> ('a2 -> 'a2 -> 'a2) -> (ad -> 'a1 -> 'a2) -> 'a1 map -> 'a2 **)

let mapFold neutral op f m =
  mapFold1 neutral op f (fun a -> a) m

(** val dmin : ('a1 -> 'a1 -> bool) -> 'a1 -> 'a1 -> 'a1 **)

let dmin dle d d' =
  match dle d d' with
  | True -> d
  | False -> d'

(** val ddmin :
    ('a1 -> 'a1 -> bool) -> 'a1 option -> 'a1 option -> 'a1 option **)

let ddmin dle dd dd' =
  match dd with
  | Some d -> (match dd' with
               | Some d' -> Some (dmin dle d d')
               | None -> dd)
  | None -> dd'

(** val ddle : ('a1 -> 'a1 -> bool) -> 'a1 option -> 'a1 option -> bool **)

let ddle dle dd dd' =
  match dd with
  | Some d -> (match dd' with
               | Some d' -> dle d d'
               | None -> True)
  | None -> (match dd' with
             | Some _ -> False
             | None -> True)

(** val ddplus : ('a1 -> 'a1 -> 'a1) -> 'a1 option -> 'a1 -> 'a1 option **)

let ddplus dplus dd d' =
  match dd with
  | Some d -> Some (dplus d d')
  | None -> dd

type 'd cGraph1 = 'd map map

(** val all_min :
    ('a1 -> 'a1 -> bool) -> (ad -> 'a1 -> 'a1 option) -> 'a1 map -> 'a1 option **)

let all_min dle f =
  mapFold None (ddmin dle) f

(** val cG_add :
    ('a1 -> 'a1 -> bool) -> 'a1 cGraph1 -> ad -> ad -> 'a1 -> 'a1 map map **)

let cG_add dle cg x y d =
  match mapGet cg x with
  | Some edges ->
    (match mapGet edges y with
     | Some d0 -> mapPut cg x (mapPut edges y (dmin dle d d0))
     | None -> mapPut cg x (mapPut edges y d))
  | None -> mapPut cg x (M1 (y, d))

(** val ad_1_path_dist_1 :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1 cGraph1 -> ad
    -> ad -> fSet -> nat -> 'a1 option **)

let rec ad_1_path_dist_1 dz dplus dle cg x y s n0 =
  match N.eqb x y with
  | True -> Some dz
  | False ->
    (match n0 with
     | O -> None
     | S n' ->
       (match mapGet cg x with
        | Some edges ->
          (match mapGet s x with
           | Some _ -> None
           | None ->
             let s' = mapPut s x Tt in
             all_min dle (fun z0 d ->
               match mapGet s' z0 with
               | Some _ -> None
               | None ->
                 ddplus dplus (ad_1_path_dist_1 dz dplus dle cg z0 y s' n') d)
               edges)
        | None -> None))

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

(** val cG_test_ineq :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
    cGraph1 -> nat -> ad -> ad -> 'a1 -> bool **)

let cG_test_ineq dz dplus dneg dle cg n0 x y d =
  ddle dle (Some (dneg d)) (ad_1_path_dist_1 dz dplus dle cg y x M0 n0)

(** val cGS_solve_1 :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) ->
    bool -> 'a1 cGraph1 -> nat -> 'a1 cGSForm list -> nat -> bool **)

let rec cGS_solve_1 dz dplus dneg dle def_answer cg n0 fsl = function
| O -> def_answer
| S gas' ->
  (match fsl with
   | Nil -> True
   | Cons (fs, fsl') ->
     (match fs with
      | CGSleq (x, y, d) ->
        (match cG_test_ineq dz dplus dneg dle cg n0 x y d with
         | True ->
           let cg' = cG_add dle cg x y d in
           cGS_solve_1 dz dplus dneg dle def_answer cg' (S n0) fsl' gas'
         | False -> False)
      | CGSand (fs0, fs1) ->
        cGS_solve_1 dz dplus dneg dle def_answer cg n0 (Cons (fs0, (Cons
          (fs1, fsl')))) gas'
      | CGSor (fs0, fs1) ->
        (match cGS_solve_1 dz dplus dneg dle def_answer cg n0 (Cons (fs0,
                 fsl')) gas' with
         | True -> True
         | False ->
           cGS_solve_1 dz dplus dneg dle def_answer cg n0 (Cons (fs1, fsl'))
             gas')))

(** val fSize : 'a1 cGSForm -> nat **)

let rec fSize = function
| CGSleq (_, _, _) -> S O
| CGSand (f0, f1) -> S (add (fSize f0) (fSize f1))
| CGSor (f0, f1) -> S (add (fSize f0) (fSize f1))

(** val cGS_solve :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) ->
    bool -> 'a1 cGSForm -> bool **)

let cGS_solve dz dplus dneg dle def_answer fs =
  cGS_solve_1 dz dplus dneg dle def_answer M0 O (Cons (fs, Nil)) (S
    (fSize fs))

(** val cGSeq : ('a1 -> 'a1) -> ad -> ad -> 'a1 -> 'a1 cGSForm **)

let cGSeq dneg x y d =
  CGSand ((CGSleq (x, y, d)), (CGSleq (y, x, (dneg d))))

(** val cGSnot :
    ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> 'a1 -> 'a1 cGSForm -> 'a1 cGSForm **)

let rec cGSnot dplus dneg done0 = function
| CGSleq (x, y, d) -> CGSleq (y, x, (dneg (dplus d done0)))
| CGSand (f0, f1) ->
  CGSor ((cGSnot dplus dneg done0 f0), (cGSnot dplus dneg done0 f1))
| CGSor (f0, f1) ->
  CGSand ((cGSnot dplus dneg done0 f0), (cGSnot dplus dneg done0 f1))

(** val cGFormSimplify :
    ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> 'a1 -> 'a1 cGForm -> 'a1 cGSForm **)

let rec cGFormSimplify dplus dneg done0 = function
| CGleq (x, y, d) -> CGSleq (x, y, d)
| CGeq (x, y, d) -> cGSeq dneg x y d
| CGand (f0, f1) ->
  CGSand ((cGFormSimplify dplus dneg done0 f0),
    (cGFormSimplify dplus dneg done0 f1))
| CGor (f0, f1) ->
  CGSor ((cGFormSimplify dplus dneg done0 f0),
    (cGFormSimplify dplus dneg done0 f1))
| CGimp (f0, f1) ->
  CGSor ((cGSnot dplus dneg done0 (cGFormSimplify dplus dneg done0 f0)),
    (cGFormSimplify dplus dneg done0 f1))
| CGnot f0 -> cGSnot dplus dneg done0 (cGFormSimplify dplus dneg done0 f0)

(** val cG_solve :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
    -> 'a1 cGForm -> bool **)

let cG_solve dz dplus dneg dle done0 f =
  cGS_solve dz dplus dneg dle False (cGFormSimplify dplus dneg done0 f)

(** val cG_prove :
    'a1 -> ('a1 -> 'a1 -> 'a1) -> ('a1 -> 'a1) -> ('a1 -> 'a1 -> bool) -> 'a1
    -> 'a1 cGForm -> bool **)

let cG_prove dz dplus dneg dle done0 f =
  negb (cG_solve dz dplus dneg dle done0 (CGnot f))

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

(** val zCGtranslate : zCGForm -> z cGForm **)

let rec zCGtranslate = function
| ZCGle (x, y) -> CGleq (x, y, Z0)
| ZCGge (x, y) -> CGleq (y, x, Z0)
| ZCGlt (x, y) -> CGleq (x, y, (Zneg XH))
| ZCGgt (x, y) -> CGleq (y, x, (Zneg XH))
| ZCGlep (x, y, k) -> CGleq (x, y, k)
| ZCGgep (x, y, k) -> CGleq (y, x, (Z.opp k))
| ZCGltp (x, y, k) -> CGleq (x, y, (Z.sub k (Zpos XH)))
| ZCGgtp (x, y, k) -> CGleq (y, x, (Z.opp (Z.add k (Zpos XH))))
| ZCGlem (x, y, k) -> CGleq (x, y, (Z.opp k))
| ZCGgem (x, y, k) -> CGleq (y, x, k)
| ZCGltm (x, y, k) -> CGleq (x, y, (Z.opp (Z.add k (Zpos XH))))
| ZCGgtm (x, y, k) -> CGleq (y, x, (Z.sub k (Zpos XH)))
| ZCGlepm (x, y, k, k') -> CGleq (x, y, (Z.sub k' k))
| ZCGgepm (x, y, k, k') -> CGleq (y, x, (Z.sub k k'))
| ZCGltpm (x, y, k, k') -> CGleq (x, y, (Z.sub (Z.sub k' k) (Zpos XH)))
| ZCGgtpm (x, y, k, k') -> CGleq (y, x, (Z.sub (Z.sub k k') (Zpos XH)))
| ZCGeq (x, y) -> CGeq (x, y, Z0)
| ZCGeqp (x, y, k) -> CGeq (x, y, k)
| ZCGeqm (x, y, k) -> CGeq (y, x, k)
| ZCGeqpm (x, y, k, k') -> CGeq (x, y, (Z.sub k' k))
| ZCGzylem (y, k) -> CGleq (N0, y, (Z.opp k))
| ZCGzygem (y, k) -> CGleq (y, N0, k)
| ZCGzyltm (y, k) -> CGleq (N0, y, (Z.opp (Z.add k (Zpos XH))))
| ZCGzygtm (y, k) -> CGleq (y, N0, (Z.sub k (Zpos XH)))
| ZCGzylepm (y, k, k') -> CGleq (N0, y, (Z.sub k' k))
| ZCGzygepm (y, k, k') -> CGleq (y, N0, (Z.sub k k'))
| ZCGzyltpm (y, k, k') -> CGleq (N0, y, (Z.sub (Z.sub k' k) (Zpos XH)))
| ZCGzygtpm (y, k, k') -> CGleq (y, N0, (Z.sub (Z.sub k k') (Zpos XH)))
| ZCGzyeqm (y, k) -> CGeq (y, N0, k)
| ZCGzyeqpm (y, k, k') -> CGeq (y, N0, (Z.sub k k'))
| ZCGxzlep (x, k) -> CGleq (x, N0, k)
| ZCGxzgep (x, k) -> CGleq (N0, x, (Z.opp k))
| ZCGxzltp (x, k) -> CGleq (x, N0, (Z.sub k (Zpos XH)))
| ZCGxzgtp (x, k) -> CGleq (N0, x, (Z.opp (Z.add k (Zpos XH))))
| ZCGxzlepm (x, k, k') -> CGleq (x, N0, (Z.sub k' k))
| ZCGxzgepm (x, k, k') -> CGleq (N0, x, (Z.sub k k'))
| ZCGxzltpm (x, k, k') -> CGleq (x, N0, (Z.sub (Z.sub k' k) (Zpos XH)))
| ZCGxzgtpm (x, k, k') -> CGleq (N0, x, (Z.sub (Z.sub k k') (Zpos XH)))
| ZCGxzeqp (x, k) -> CGeq (x, N0, k)
| ZCGxzeqpm (x, k, k') -> CGeq (x, N0, (Z.sub k' k))
| ZCGzzlep (k, k') -> CGleq (N0, N0, (Z.sub k' k))
| ZCGzzltp (k, k') -> CGleq (N0, N0, (Z.sub (Z.sub k' k) (Zpos XH)))
| ZCGzzgep (k, k') -> CGleq (N0, N0, (Z.sub k k'))
| ZCGzzgtp (k, k') -> CGleq (N0, N0, (Z.sub (Z.sub k k') (Zpos XH)))
| ZCGzzeq (k, k') -> CGeq (N0, N0, (Z.sub k k'))
| ZCGand (f0, f1) -> CGand ((zCGtranslate f0), (zCGtranslate f1))
| ZCGor (f0, f1) -> CGor ((zCGtranslate f0), (zCGtranslate f1))
| ZCGimp (f0, f1) -> CGimp ((zCGtranslate f0), (zCGtranslate f1))
| ZCGnot f0 -> CGnot (zCGtranslate f0)
| ZCGiff (f0, f1) ->
  CGand ((CGimp ((zCGtranslate f0), (zCGtranslate f1))), (CGimp
    ((zCGtranslate f1), (zCGtranslate f0))))

(** val zCG_prove : zCGForm -> bool **)

let zCG_prove f =
  cG_prove Z0 Z.add Z.opp Z.leb (Zpos XH) (zCGtranslate f)

(** val i2p : int -> positive **)

let i2p = 
  let rec i2p = function 
    1 -> XH 
  | n -> let n' = i2p (n/2) in if (n mod 2)=0 then XO n' else XI n'
  in i2p


(** val i2a : int -> ad **)

let i2a =  function 
    0 -> N0
  | n -> Npos (i2p n)


(** val i2z : int -> z **)

let i2z =  function
    0 -> Z0
  | n -> if n < 0 then Zneg (i2p (-n)) else Zpos (i2p n)

