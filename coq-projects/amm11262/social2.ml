
type __ = Obj.t
let __ = let rec f _ = Obj.repr f in Obj.repr f

type unit0 =
| Tt

type bool =
| True
| False

type nat =
| O
| S of nat

type 'a option =
| Some of 'a
| None

type ('a, 'b) prod =
| Pair of 'a * 'b

(** val fst : ('a1, 'a2) prod -> 'a1 **)

let fst = function
| Pair (x, _) -> x

(** val snd : ('a1, 'a2) prod -> 'a2 **)

let snd = function
| Pair (_, y) -> y

type 'a list =
| Nil
| Cons of 'a * 'a list

(** val length : 'a1 list -> nat **)

let rec length = function
| Nil -> O
| Cons (_, l') -> S (length l')

(** val app : 'a1 list -> 'a1 list -> 'a1 list **)

let rec app l m =
  match l with
  | Nil -> m
  | Cons (a, l1) -> Cons (a, (app l1 m))

type comparison =
| Eq
| Lt
| Gt

type compareSpecT =
| CompEqT
| CompLtT
| CompGtT

(** val compareSpec2Type : comparison -> compareSpecT **)

let compareSpec2Type = function
| Eq -> CompEqT
| Lt -> CompLtT
| Gt -> CompGtT

type 'a compSpecT = compareSpecT

(** val compSpec2Type : 'a1 -> 'a1 -> comparison -> 'a1 compSpecT **)

let compSpec2Type _ _ =
  compareSpec2Type

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type sumbool =
| Left
| Right

type 'a sumor =
| Inleft of 'a
| Inright

(** val pred : nat -> nat **)

let pred n = match n with
| O -> n
| S u -> u

(** val flip : ('a1 -> 'a2 -> 'a3) -> 'a2 -> 'a1 -> 'a3 **)

let flip f x y =
  f y x

module type DecidableTypeOrig =
 sig
  type t

  val eq_dec : t -> t -> sumbool
 end

module type OrderedType =
 sig
  type t

  val compare : t -> t -> comparison

  val eq_dec : t -> t -> sumbool
 end

module type OrderedType' =
 sig
  type t

  val compare : t -> t -> comparison

  val eq_dec : t -> t -> sumbool
 end

module OT_to_Full =
 functor (O:OrderedType') ->
 struct
  type t = O.t

  (** val compare : t -> t -> comparison **)

  let compare =
    O.compare

  (** val eq_dec : t -> t -> sumbool **)

  let eq_dec =
    O.eq_dec
 end

module OT_to_OrderTac =
 functor (OT:OrderedType) ->
 struct
  module OTF = OT_to_Full(OT)

  module TO =
   struct
    type t = OTF.t

    (** val compare : t -> t -> comparison **)

    let compare =
      OTF.compare

    (** val eq_dec : t -> t -> sumbool **)

    let eq_dec =
      OTF.eq_dec
   end
 end

module OrderedTypeFacts =
 functor (O:OrderedType') ->
 struct
  module OrderTac = OT_to_OrderTac(O)

  (** val eq_dec : O.t -> O.t -> sumbool **)

  let eq_dec =
    O.eq_dec

  (** val lt_dec : O.t -> O.t -> sumbool **)

  let lt_dec x y =
    let c = compSpec2Type x y (O.compare x y) in
    (match c with
     | CompLtT -> Left
     | _ -> Right)

  (** val eqb : O.t -> O.t -> bool **)

  let eqb x y =
    match eq_dec x y with
    | Left -> True
    | Right -> False
 end

module Nat =
 struct
  (** val compare : nat -> nat -> comparison **)

  let rec compare n m =
    match n with
    | O -> (match m with
            | O -> Eq
            | S _ -> Lt)
    | S n' -> (match m with
               | O -> Gt
               | S m' -> compare n' m')

  (** val eq_dec : nat -> nat -> sumbool **)

  let rec eq_dec n m =
    match n with
    | O -> (match m with
            | O -> Left
            | S _ -> Right)
    | S n0 -> (match m with
               | O -> Right
               | S m0 -> eq_dec n0 m0)
 end

(** val lt_eq_lt_dec : nat -> nat -> sumbool sumor **)

let rec lt_eq_lt_dec n m =
  match n with
  | O -> (match m with
          | O -> Inleft Right
          | S _ -> Inleft Left)
  | S n0 -> (match m with
             | O -> Inright
             | S m0 -> lt_eq_lt_dec n0 m0)

(** val le_lt_eq_dec : nat -> nat -> sumbool **)

let le_lt_eq_dec n m =
  let s = lt_eq_lt_dec n m in
  (match s with
   | Inleft s0 -> s0
   | Inright -> assert false (* absurd case *))

(** val rev : 'a1 list -> 'a1 list **)

let rec rev = function
| Nil -> Nil
| Cons (x, l') -> app (rev l') (Cons (x, Nil))

(** val fold_left : ('a1 -> 'a2 -> 'a1) -> 'a2 list -> 'a1 -> 'a1 **)

let rec fold_left f l a0 =
  match l with
  | Nil -> a0
  | Cons (b, t0) -> fold_left f t0 (f a0 b)

(** val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1 **)

let rec fold_right f a0 = function
| Nil -> a0
| Cons (b, t0) -> f b (fold_right f a0 t0)

type 'x compare0 =
| LT
| EQ
| GT

module type Coq_OrderedType =
 sig
  type t

  val compare : t -> t -> t compare0

  val eq_dec : t -> t -> sumbool
 end

module Nat_as_OT =
 struct
  type t = nat

  (** val compare : nat -> nat -> nat compare0 **)

  let compare x y =
    match Nat.compare x y with
    | Eq -> EQ
    | Lt -> LT
    | Gt -> GT

  (** val eq_dec : nat -> nat -> sumbool **)

  let eq_dec =
    Nat.eq_dec
 end

module type DecidableType =
 DecidableTypeOrig

module type WS =
 sig
  module E :
   DecidableType

  type elt = E.t

  type t

  val empty : t

  val is_empty : t -> bool

  val mem : elt -> t -> bool

  val add : elt -> t -> t

  val singleton : elt -> t

  val remove : elt -> t -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val eq_dec : t -> t -> sumbool

  val equal : t -> t -> bool

  val subset : t -> t -> bool

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val filter : (elt -> bool) -> t -> t

  val partition : (elt -> bool) -> t -> (t, t) prod

  val cardinal : t -> nat

  val elements : t -> elt list

  val choose : t -> elt option
 end

module WFacts_fun =
 functor (E:DecidableType) ->
 functor (M:sig
  type elt = E.t

  type t

  val empty : t

  val is_empty : t -> bool

  val mem : elt -> t -> bool

  val add : elt -> t -> t

  val singleton : elt -> t

  val remove : elt -> t -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val eq_dec : t -> t -> sumbool

  val equal : t -> t -> bool

  val subset : t -> t -> bool

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val filter : (elt -> bool) -> t -> t

  val partition : (elt -> bool) -> t -> (t, t) prod

  val cardinal : t -> nat

  val elements : t -> elt list

  val choose : t -> elt option
 end) ->
 struct
  (** val eqb : E.t -> E.t -> bool **)

  let eqb x y =
    match E.eq_dec x y with
    | Left -> True
    | Right -> False
 end

module MakeListOrdering =
 functor (O:OrderedType) ->
 struct
  module MO = OrderedTypeFacts(O)
 end

module OrderedTypeLists =
 functor (O:OrderedType) ->
 struct
 end

module MakeRaw =
 functor (X:OrderedType) ->
 struct
  module MX = OrderedTypeFacts(X)

  module ML = OrderedTypeLists(X)

  type elt = X.t

  type t = elt list

  (** val empty : t **)

  let empty =
    Nil

  (** val is_empty : t -> bool **)

  let is_empty = function
  | Nil -> True
  | Cons (_, _) -> False

  (** val mem : X.t -> X.t list -> bool **)

  let rec mem x = function
  | Nil -> False
  | Cons (y, l) ->
    (match X.compare x y with
     | Eq -> True
     | Lt -> False
     | Gt -> mem x l)

  (** val add : X.t -> X.t list -> X.t list **)

  let rec add x s = match s with
  | Nil -> Cons (x, Nil)
  | Cons (y, l) ->
    (match X.compare x y with
     | Eq -> s
     | Lt -> Cons (x, s)
     | Gt -> Cons (y, (add x l)))

  (** val singleton : elt -> elt list **)

  let singleton x =
    Cons (x, Nil)

  (** val remove : X.t -> X.t list -> t **)

  let rec remove x s = match s with
  | Nil -> Nil
  | Cons (y, l) ->
    (match X.compare x y with
     | Eq -> l
     | Lt -> s
     | Gt -> Cons (y, (remove x l)))

  (** val union : t -> t -> t **)

  let rec union s = match s with
  | Nil -> (fun s' -> s')
  | Cons (x, l) ->
    let rec union_aux s' = match s' with
    | Nil -> s
    | Cons (x', l') ->
      (match X.compare x x' with
       | Eq -> Cons (x, (union l l'))
       | Lt -> Cons (x, (union l s'))
       | Gt -> Cons (x', (union_aux l')))
    in union_aux

  (** val inter : t -> t -> t **)

  let rec inter = function
  | Nil -> (fun _ -> Nil)
  | Cons (x, l) ->
    let rec inter_aux s' = match s' with
    | Nil -> Nil
    | Cons (x', l') ->
      (match X.compare x x' with
       | Eq -> Cons (x, (inter l l'))
       | Lt -> inter l s'
       | Gt -> inter_aux l')
    in inter_aux

  (** val diff : t -> t -> t **)

  let rec diff s = match s with
  | Nil -> (fun _ -> Nil)
  | Cons (x, l) ->
    let rec diff_aux s' = match s' with
    | Nil -> s
    | Cons (x', l') ->
      (match X.compare x x' with
       | Eq -> diff l l'
       | Lt -> Cons (x, (diff l s'))
       | Gt -> diff_aux l')
    in diff_aux

  (** val equal : t -> t -> bool **)

  let rec equal s s' =
    match s with
    | Nil -> (match s' with
              | Nil -> True
              | Cons (_, _) -> False)
    | Cons (x, l) ->
      (match s' with
       | Nil -> False
       | Cons (x', l') ->
         (match X.compare x x' with
          | Eq -> equal l l'
          | _ -> False))

  (** val subset : X.t list -> X.t list -> bool **)

  let rec subset s s' =
    match s with
    | Nil -> True
    | Cons (x, l) ->
      (match s' with
       | Nil -> False
       | Cons (x', l') ->
         (match X.compare x x' with
          | Eq -> subset l l'
          | Lt -> False
          | Gt -> subset s l'))

  (** val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1 **)

  let fold f s i =
    fold_left (flip f) s i

  (** val filter : (elt -> bool) -> t -> t **)

  let rec filter f = function
  | Nil -> Nil
  | Cons (x, l) ->
    (match f x with
     | True -> Cons (x, (filter f l))
     | False -> filter f l)

  (** val for_all : (elt -> bool) -> t -> bool **)

  let rec for_all f = function
  | Nil -> True
  | Cons (x, l) -> (match f x with
                    | True -> for_all f l
                    | False -> False)

  (** val exists_ : (elt -> bool) -> t -> bool **)

  let rec exists_ f = function
  | Nil -> False
  | Cons (x, l) -> (match f x with
                    | True -> True
                    | False -> exists_ f l)

  (** val partition : (elt -> bool) -> t -> (t, t) prod **)

  let rec partition f = function
  | Nil -> Pair (Nil, Nil)
  | Cons (x, l) ->
    let Pair (s1, s2) = partition f l in
    (match f x with
     | True -> Pair ((Cons (x, s1)), s2)
     | False -> Pair (s1, (Cons (x, s2))))

  (** val cardinal : t -> nat **)

  let cardinal =
    length

  (** val elements : t -> elt list **)

  let elements x =
    x

  (** val min_elt : t -> elt option **)

  let min_elt = function
  | Nil -> None
  | Cons (x, _) -> Some x

  (** val max_elt : t -> elt option **)

  let rec max_elt = function
  | Nil -> None
  | Cons (x, l) -> (match l with
                    | Nil -> Some x
                    | Cons (_, _) -> max_elt l)

  (** val choose : t -> elt option **)

  let choose =
    min_elt

  (** val compare : X.t list -> X.t list -> comparison **)

  let rec compare s s' =
    match s with
    | Nil -> (match s' with
              | Nil -> Eq
              | Cons (_, _) -> Lt)
    | Cons (x, s0) ->
      (match s' with
       | Nil -> Gt
       | Cons (x', s'0) ->
         (match X.compare x x' with
          | Eq -> compare s0 s'0
          | x0 -> x0))

  (** val inf : X.t -> X.t list -> bool **)

  let inf x = function
  | Nil -> True
  | Cons (y, _) -> (match X.compare x y with
                    | Lt -> True
                    | _ -> False)

  (** val isok : X.t list -> bool **)

  let rec isok = function
  | Nil -> True
  | Cons (x, l0) -> (match inf x l0 with
                     | True -> isok l0
                     | False -> False)

  module L = MakeListOrdering(X)
 end

module Make =
 functor (X:OrderedType) ->
 struct
  module Raw = MakeRaw(X)

  module E =
   struct
    type t = X.t

    (** val compare : t -> t -> comparison **)

    let compare =
      X.compare

    (** val eq_dec : t -> t -> sumbool **)

    let eq_dec =
      X.eq_dec
   end

  type elt = X.t

  type t_ = Raw.t
    (* singleton inductive, whose constructor was Mkt *)

  (** val this : t_ -> Raw.t **)

  let this t0 =
    t0

  type t = t_

  (** val mem : elt -> t -> bool **)

  let mem x s =
    Raw.mem x (this s)

  (** val add : elt -> t -> t **)

  let add x s =
    Raw.add x (this s)

  (** val remove : elt -> t -> t **)

  let remove x s =
    Raw.remove x (this s)

  (** val singleton : elt -> t **)

  let singleton =
    Raw.singleton

  (** val union : t -> t -> t **)

  let union s s' =
    Raw.union (this s) (this s')

  (** val inter : t -> t -> t **)

  let inter s s' =
    Raw.inter (this s) (this s')

  (** val diff : t -> t -> t **)

  let diff s s' =
    Raw.diff (this s) (this s')

  (** val equal : t -> t -> bool **)

  let equal s s' =
    Raw.equal (this s) (this s')

  (** val subset : t -> t -> bool **)

  let subset s s' =
    Raw.subset (this s) (this s')

  (** val empty : t **)

  let empty =
    Raw.empty

  (** val is_empty : t -> bool **)

  let is_empty s =
    Raw.is_empty (this s)

  (** val elements : t -> elt list **)

  let elements s =
    Raw.elements (this s)

  (** val choose : t -> elt option **)

  let choose s =
    Raw.choose (this s)

  (** val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1 **)

  let fold f s =
    Raw.fold f (this s)

  (** val cardinal : t -> nat **)

  let cardinal s =
    Raw.cardinal (this s)

  (** val filter : (elt -> bool) -> t -> t **)

  let filter f s =
    Raw.filter f (this s)

  (** val for_all : (elt -> bool) -> t -> bool **)

  let for_all f s =
    Raw.for_all f (this s)

  (** val exists_ : (elt -> bool) -> t -> bool **)

  let exists_ f s =
    Raw.exists_ f (this s)

  (** val partition : (elt -> bool) -> t -> (t, t) prod **)

  let partition f s =
    let p = Raw.partition f (this s) in Pair ((fst p), (snd p))

  (** val eq_dec : t -> t -> sumbool **)

  let eq_dec s0 s'0 =
    let b = Raw.equal s0 s'0 in (match b with
                                 | True -> Left
                                 | False -> Right)

  (** val compare : t -> t -> comparison **)

  let compare s s' =
    Raw.compare (this s) (this s')

  (** val min_elt : t -> elt option **)

  let min_elt s =
    Raw.min_elt (this s)

  (** val max_elt : t -> elt option **)

  let max_elt s =
    Raw.max_elt (this s)
 end

module type OrderedTypeOrig =
 Coq_OrderedType

module Update_OT =
 functor (O:OrderedTypeOrig) ->
 struct
  type t = O.t

  (** val eq_dec : t -> t -> sumbool **)

  let eq_dec =
    O.eq_dec

  (** val compare : O.t -> O.t -> comparison **)

  let compare x y =
    match O.compare x y with
    | LT -> Lt
    | EQ -> Eq
    | GT -> Gt
 end

module Coq_Make =
 functor (X:Coq_OrderedType) ->
 struct
  module X' = Update_OT(X)

  module MSet = Make(X')

  type elt = X.t

  type t = MSet.t

  (** val empty : t **)

  let empty =
    MSet.empty

  (** val is_empty : t -> bool **)

  let is_empty =
    MSet.is_empty

  (** val mem : elt -> t -> bool **)

  let mem =
    MSet.mem

  (** val add : elt -> t -> t **)

  let add =
    MSet.add

  (** val singleton : elt -> t **)

  let singleton =
    MSet.singleton

  (** val remove : elt -> t -> t **)

  let remove =
    MSet.remove

  (** val union : t -> t -> t **)

  let union =
    MSet.union

  (** val inter : t -> t -> t **)

  let inter =
    MSet.inter

  (** val diff : t -> t -> t **)

  let diff =
    MSet.diff

  (** val eq_dec : t -> t -> sumbool **)

  let eq_dec =
    MSet.eq_dec

  (** val equal : t -> t -> bool **)

  let equal =
    MSet.equal

  (** val subset : t -> t -> bool **)

  let subset =
    MSet.subset

  (** val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1 **)

  let fold =
    MSet.fold

  (** val for_all : (elt -> bool) -> t -> bool **)

  let for_all =
    MSet.for_all

  (** val exists_ : (elt -> bool) -> t -> bool **)

  let exists_ =
    MSet.exists_

  (** val filter : (elt -> bool) -> t -> t **)

  let filter =
    MSet.filter

  (** val partition : (elt -> bool) -> t -> (t, t) prod **)

  let partition =
    MSet.partition

  (** val cardinal : t -> nat **)

  let cardinal =
    MSet.cardinal

  (** val elements : t -> elt list **)

  let elements =
    MSet.elements

  (** val choose : t -> elt option **)

  let choose =
    MSet.choose

  module MF =
   struct
    (** val eqb : X.t -> X.t -> bool **)

    let eqb x y =
      match MSet.E.eq_dec x y with
      | Left -> True
      | Right -> False
   end

  (** val min_elt : t -> elt option **)

  let min_elt =
    MSet.min_elt

  (** val max_elt : t -> elt option **)

  let max_elt =
    MSet.max_elt

  (** val compare : t -> t -> t compare0 **)

  let compare s s' =
    let c = compSpec2Type s s' (MSet.compare s s') in
    (match c with
     | CompEqT -> EQ
     | CompLtT -> LT
     | CompGtT -> GT)

  module E =
   struct
    type t = X.t

    (** val compare : t -> t -> t compare0 **)

    let compare =
      X.compare

    (** val eq_dec : t -> t -> sumbool **)

    let eq_dec =
      X.eq_dec
   end
 end

module WDecide_fun =
 functor (E:DecidableType) ->
 functor (M:sig
  type elt = E.t

  type t

  val empty : t

  val is_empty : t -> bool

  val mem : elt -> t -> bool

  val add : elt -> t -> t

  val singleton : elt -> t

  val remove : elt -> t -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val eq_dec : t -> t -> sumbool

  val equal : t -> t -> bool

  val subset : t -> t -> bool

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val filter : (elt -> bool) -> t -> t

  val partition : (elt -> bool) -> t -> (t, t) prod

  val cardinal : t -> nat

  val elements : t -> elt list

  val choose : t -> elt option
 end) ->
 struct
  module F = WFacts_fun(E)(M)

  module FSetLogicalFacts =
   struct
   end

  module FSetDecideAuxiliary =
   struct
   end

  module FSetDecideTestCases =
   struct
   end
 end

module WProperties_fun =
 functor (E:DecidableType) ->
 functor (M:sig
  type elt = E.t

  type t

  val empty : t

  val is_empty : t -> bool

  val mem : elt -> t -> bool

  val add : elt -> t -> t

  val singleton : elt -> t

  val remove : elt -> t -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val eq_dec : t -> t -> sumbool

  val equal : t -> t -> bool

  val subset : t -> t -> bool

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val filter : (elt -> bool) -> t -> t

  val partition : (elt -> bool) -> t -> (t, t) prod

  val cardinal : t -> nat

  val elements : t -> elt list

  val choose : t -> elt option
 end) ->
 struct
  module Dec = WDecide_fun(E)(M)

  module FM = Dec.F

  (** val coq_In_dec : M.elt -> M.t -> sumbool **)

  let coq_In_dec x s =
    match M.mem x s with
    | True -> Left
    | False -> Right

  (** val of_list : M.elt list -> M.t **)

  let of_list l =
    fold_right M.add M.empty l

  (** val to_list : M.t -> M.elt list **)

  let to_list =
    M.elements

  (** val fold_rec :
      (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> __ -> 'a2) -> (M.elt ->
      'a1 -> M.t -> M.t -> __ -> __ -> __ -> 'a2 -> 'a2) -> 'a2 **)

  let fold_rec f i s pempty pstep =
    let l = rev (M.elements s) in
    let pstep' = fun x a s' s'' x0 -> pstep x a s' s'' __ __ __ x0 in
    let rec f0 l0 pstep'0 s0 =
      match l0 with
      | Nil -> pempty s0 __
      | Cons (y, l1) ->
        pstep'0 y (fold_right f i l1) (of_list l1) s0 __ __ __
          (f0 l1 (fun x a0 s' s'' _ _ _ x0 ->
            pstep'0 x a0 s' s'' __ __ __ x0) (of_list l1))
    in f0 l (fun x a s' s'' _ _ _ x0 -> pstep' x a s' s'' x0) s

  (** val fold_rec_bis :
      (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> M.t -> 'a1 -> __ -> 'a2
      -> 'a2) -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> __ -> 'a2 -> 'a2) -> 'a2 **)

  let fold_rec_bis f i s pmorphism pempty pstep =
    fold_rec f i s (fun s' _ -> pmorphism M.empty s' i __ pempty)
      (fun x a s' s'' _ _ _ x0 ->
      pmorphism (M.add x s') s'' (f x a) __ (pstep x a s' __ __ x0))

  (** val fold_rec_nodep :
      (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> 'a2 -> (M.elt -> 'a1 -> __ ->
      'a2 -> 'a2) -> 'a2 **)

  let fold_rec_nodep f i s x x0 =
    fold_rec_bis f i s (fun _ _ _ _ x1 -> x1) x (fun x1 a _ _ _ x2 ->
      x0 x1 a __ x2)

  (** val fold_rec_weak :
      (M.elt -> 'a1 -> 'a1) -> 'a1 -> (M.t -> M.t -> 'a1 -> __ -> 'a2 -> 'a2)
      -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> 'a2 -> 'a2) -> M.t -> 'a2 **)

  let fold_rec_weak f i x x0 x1 s =
    fold_rec_bis f i s x x0 (fun x2 a s' _ _ x3 -> x1 x2 a s' __ x3)

  (** val fold_rel :
      (M.elt -> 'a1 -> 'a1) -> (M.elt -> 'a2 -> 'a2) -> 'a1 -> 'a2 -> M.t ->
      'a3 -> (M.elt -> 'a1 -> 'a2 -> __ -> 'a3 -> 'a3) -> 'a3 **)

  let fold_rel f g i j s rempty rstep =
    let l = rev (M.elements s) in
    let rstep' = fun x a b x0 -> rstep x a b __ x0 in
    let rec f0 l0 rstep'0 =
      match l0 with
      | Nil -> rempty
      | Cons (y, l1) ->
        rstep'0 y (fold_right f i l1) (fold_right g j l1) __
          (f0 l1 (fun x a0 b _ x0 -> rstep'0 x a0 b __ x0))
    in f0 l (fun x a b _ x0 -> rstep' x a b x0)

  (** val set_induction :
      (M.t -> __ -> 'a1) -> (M.t -> M.t -> 'a1 -> M.elt -> __ -> __ -> 'a1)
      -> M.t -> 'a1 **)

  let set_induction x x0 s =
    fold_rec (fun _ _ -> Tt) Tt s x (fun x1 _ s' s'' _ _ _ x2 ->
      x0 s' s'' x2 x1 __ __)

  (** val set_induction_bis :
      (M.t -> M.t -> __ -> 'a1 -> 'a1) -> 'a1 -> (M.elt -> M.t -> __ -> 'a1
      -> 'a1) -> M.t -> 'a1 **)

  let set_induction_bis x x0 x1 s =
    fold_rec_bis (fun _ _ -> Tt) Tt s (fun s0 s' _ _ x2 -> x s0 s' __ x2) x0
      (fun x2 _ s' _ _ x3 -> x1 x2 s' __ x3)

  (** val cardinal_inv_2 : M.t -> nat -> M.elt **)

  let cardinal_inv_2 s _ =
    let l = M.elements s in
    (match l with
     | Nil -> assert false (* absurd case *)
     | Cons (e, _) -> e)

  (** val cardinal_inv_2b : M.t -> M.elt **)

  let cardinal_inv_2b s =
    let n = M.cardinal s in
    let x = fun x -> cardinal_inv_2 s x in
    (match n with
     | O -> assert false (* absurd case *)
     | S n0 -> x n0)
 end

module WProperties =
 functor (M:WS) ->
 WProperties_fun(M.E)(M)

module Properties = WProperties

module NatSet = Coq_Make(Nat_as_OT)

module GeneralProperties = Properties(NatSet)

(** val extendible_by_one : NatSet.t -> nat -> NatSet.t -> NatSet.elt **)

let extendible_by_one town _ b' =
  GeneralProperties.cardinal_inv_2 (NatSet.diff town b')
    (pred (NatSet.cardinal (NatSet.diff town b')))

(** val extendible_to_n : NatSet.t -> nat -> NatSet.t -> NatSet.t **)

let extendible_to_n town n b' =
  let rec f = function
  | O -> NatSet.empty
  | S n1 ->
    let s = le_lt_eq_dec (NatSet.cardinal b') (S n1) in
    (match s with
     | Left ->
       let s0 = f n1 in
       let s1 = extendible_by_one town n s0 in NatSet.add s1 s0
     | Right -> b')
  in f n

(** val inductive_invariant :
    NatSet.t -> nat -> (NatSet.t -> __ -> __ -> NatSet.elt) -> nat -> NatSet.t **)

let rec inductive_invariant town n property = function
| O -> NatSet.empty
| S n0 ->
  let s = inductive_invariant town n property n0 in
  let s0 = extendible_to_n town n s in
  let s1 = property s0 __ __ in NatSet.add s1 s

(** val aMM11262 :
    NatSet.t -> nat -> (NatSet.t -> __ -> __ -> NatSet.elt) -> NatSet.elt **)

let aMM11262 town n property =
  let s = inductive_invariant town n property n in
  let s0 = property s __ __ in
  let c = NatSet.diff town (NatSet.add s0 s) in property c __ __

(** val town_2 : NatSet.t **)

let town_2 =
  NatSet.add (S O)
    (NatSet.add (S (S O))
      (NatSet.add (S (S (S O)))
        (NatSet.add (S (S (S (S O))))
          (NatSet.add (S (S (S (S (S O))))) NatSet.empty))))

(** val subsets_2 :
    NatSet.t -> sumbool sumor sumor sumor sumor sumor sumor sumor sumor **)

let subsets_2 b =
  let s = GeneralProperties.coq_In_dec (S O) b in
  (match s with
   | Left ->
     Inleft (Inleft (Inleft (Inleft (Inleft (Inleft
       (let s0 =
          GeneralProperties.coq_In_dec (S (S O)) (NatSet.remove (S O) b)
        in
        match s0 with
        | Left -> Inleft (Inleft Left)
        | Right ->
          let s1 =
            GeneralProperties.coq_In_dec (S (S (S O))) (NatSet.remove (S O) b)
          in
          (match s1 with
           | Left -> Inleft (Inleft Right)
           | Right ->
             let s2 =
               GeneralProperties.coq_In_dec (S (S (S (S O))))
                 (NatSet.remove (S O) b)
             in
             (match s2 with
              | Left -> Inleft Inright
              | Right ->
                let s3 =
                  GeneralProperties.coq_In_dec (S (S (S (S (S O)))))
                    (NatSet.remove (S O) b)
                in
                (match s3 with
                 | Left -> Inright
                 | Right -> assert false (* absurd case *))))))))))
   | Right ->
     let s0 = GeneralProperties.coq_In_dec (S (S O)) b in
     (match s0 with
      | Left ->
        let s1 =
          GeneralProperties.coq_In_dec (S (S (S O)))
            (NatSet.remove (S (S O)) b)
        in
        (match s1 with
         | Left -> Inleft (Inleft (Inleft (Inleft (Inleft Inright))))
         | Right ->
           let s2 =
             GeneralProperties.coq_In_dec (S (S (S (S O))))
               (NatSet.remove (S (S O)) b)
           in
           (match s2 with
            | Left -> Inleft (Inleft (Inleft (Inleft Inright)))
            | Right ->
              let s3 =
                GeneralProperties.coq_In_dec (S (S (S (S (S O)))))
                  (NatSet.remove (S (S O)) b)
              in
              (match s3 with
               | Left -> Inleft (Inleft (Inleft Inright))
               | Right -> assert false (* absurd case *))))
      | Right ->
        let s1 = GeneralProperties.coq_In_dec (S (S (S O))) b in
        (match s1 with
         | Left ->
           let s2 =
             GeneralProperties.coq_In_dec (S (S (S (S O))))
               (NatSet.remove (S (S (S O))) b)
           in
           (match s2 with
            | Left -> Inleft (Inleft Inright)
            | Right ->
              let s3 =
                GeneralProperties.coq_In_dec (S (S (S (S (S O)))))
                  (NatSet.remove (S (S (S O))) b)
              in
              (match s3 with
               | Left -> Inleft Inright
               | Right -> assert false (* absurd case *)))
         | Right ->
           let s2 = GeneralProperties.coq_In_dec (S (S (S (S O)))) b in
           (match s2 with
            | Left ->
              let s3 =
                GeneralProperties.coq_In_dec (S (S (S (S (S O)))))
                  (NatSet.remove (S (S (S (S O)))) b)
              in
              (match s3 with
               | Left -> Inright
               | Right -> assert false (* absurd case *))
            | Right -> assert false (* absurd case *)))))

(** val acquintance_2 : NatSet.t -> NatSet.elt **)

let acquintance_2 b =
  let s = subsets_2 b in
  (match s with
   | Inleft s0 ->
     (match s0 with
      | Inleft s1 ->
        (match s1 with
         | Inleft s2 ->
           (match s2 with
            | Inleft s3 ->
              (match s3 with
               | Inleft s4 ->
                 (match s4 with
                  | Inleft s5 ->
                    (match s5 with
                     | Inleft s6 ->
                       (match s6 with
                        | Inleft s7 ->
                          (match s7 with
                           | Left -> S (S (S (S (S O))))
                           | Right -> S (S (S (S O))))
                        | Inright -> S (S (S O)))
                     | Inright -> S (S O))
                  | Inright -> S (S (S (S (S O)))))
               | Inright -> S O)
            | Inright -> S O)
         | Inright -> S O)
      | Inright -> S O)
   | Inright -> S (S (S O)))

(** val social_citizen_2 : NatSet.elt **)

let social_citizen_2 =
  aMM11262 town_2 (S (S O)) (fun x _ _ -> acquintance_2 x)
