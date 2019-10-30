
type __ = Obj.t

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

val fst : ('a1, 'a2) prod -> 'a1

val snd : ('a1, 'a2) prod -> 'a2

type 'a list =
| Nil
| Cons of 'a * 'a list

val length : 'a1 list -> nat

val app : 'a1 list -> 'a1 list -> 'a1 list

type comparison =
| Eq
| Lt
| Gt

type compareSpecT =
| CompEqT
| CompLtT
| CompGtT

val compareSpec2Type : comparison -> compareSpecT

type 'a compSpecT = compareSpecT

val compSpec2Type : 'a1 -> 'a1 -> comparison -> 'a1 compSpecT

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type sumbool =
| Left
| Right

type 'a sumor =
| Inleft of 'a
| Inright

val pred : nat -> nat

val flip : ('a1 -> 'a2 -> 'a3) -> 'a2 -> 'a1 -> 'a3

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

module OT_to_Full :
 functor (O:OrderedType') ->
 sig
  type t = O.t

  val compare : t -> t -> comparison

  val eq_dec : t -> t -> sumbool
 end

module OT_to_OrderTac :
 functor (OT:OrderedType) ->
 sig
  module OTF :
   sig
    type t = OT.t

    val compare : t -> t -> comparison

    val eq_dec : t -> t -> sumbool
   end

  module TO :
   sig
    type t = OTF.t

    val compare : t -> t -> comparison

    val eq_dec : t -> t -> sumbool
   end
 end

module OrderedTypeFacts :
 functor (O:OrderedType') ->
 sig
  module OrderTac :
   sig
    module OTF :
     sig
      type t = O.t

      val compare : t -> t -> comparison

      val eq_dec : t -> t -> sumbool
     end

    module TO :
     sig
      type t = OTF.t

      val compare : t -> t -> comparison

      val eq_dec : t -> t -> sumbool
     end
   end

  val eq_dec : O.t -> O.t -> sumbool

  val lt_dec : O.t -> O.t -> sumbool

  val eqb : O.t -> O.t -> bool
 end

module Nat :
 sig
  val compare : nat -> nat -> comparison

  val eq_dec : nat -> nat -> sumbool
 end

val lt_eq_lt_dec : nat -> nat -> sumbool sumor

val le_lt_eq_dec : nat -> nat -> sumbool

val rev : 'a1 list -> 'a1 list

val fold_left : ('a1 -> 'a2 -> 'a1) -> 'a2 list -> 'a1 -> 'a1

val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1

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

module Nat_as_OT :
 sig
  type t = nat

  val compare : nat -> nat -> nat compare0

  val eq_dec : nat -> nat -> sumbool
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

module WFacts_fun :
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
 sig
  val eqb : E.t -> E.t -> bool
 end

module MakeListOrdering :
 functor (O:OrderedType) ->
 sig
  module MO :
   sig
    module OrderTac :
     sig
      module OTF :
       sig
        type t = O.t

        val compare : t -> t -> comparison

        val eq_dec : t -> t -> sumbool
       end

      module TO :
       sig
        type t = OTF.t

        val compare : t -> t -> comparison

        val eq_dec : t -> t -> sumbool
       end
     end

    val eq_dec : O.t -> O.t -> sumbool

    val lt_dec : O.t -> O.t -> sumbool

    val eqb : O.t -> O.t -> bool
   end
 end

module OrderedTypeLists :
 functor (O:OrderedType) ->
 sig
 end

module MakeRaw :
 functor (X:OrderedType) ->
 sig
  module MX :
   sig
    module OrderTac :
     sig
      module OTF :
       sig
        type t = X.t

        val compare : t -> t -> comparison

        val eq_dec : t -> t -> sumbool
       end

      module TO :
       sig
        type t = OTF.t

        val compare : t -> t -> comparison

        val eq_dec : t -> t -> sumbool
       end
     end

    val eq_dec : X.t -> X.t -> sumbool

    val lt_dec : X.t -> X.t -> sumbool

    val eqb : X.t -> X.t -> bool
   end

  module ML :
   sig
   end

  type elt = X.t

  type t = elt list

  val empty : t

  val is_empty : t -> bool

  val mem : X.t -> X.t list -> bool

  val add : X.t -> X.t list -> X.t list

  val singleton : elt -> elt list

  val remove : X.t -> X.t list -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val equal : t -> t -> bool

  val subset : X.t list -> X.t list -> bool

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val filter : (elt -> bool) -> t -> t

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val partition : (elt -> bool) -> t -> (t, t) prod

  val cardinal : t -> nat

  val elements : t -> elt list

  val min_elt : t -> elt option

  val max_elt : t -> elt option

  val choose : t -> elt option

  val compare : X.t list -> X.t list -> comparison

  val inf : X.t -> X.t list -> bool

  val isok : X.t list -> bool

  module L :
   sig
    module MO :
     sig
      module OrderTac :
       sig
        module OTF :
         sig
          type t = X.t

          val compare : t -> t -> comparison

          val eq_dec : t -> t -> sumbool
         end

        module TO :
         sig
          type t = OTF.t

          val compare : t -> t -> comparison

          val eq_dec : t -> t -> sumbool
         end
       end

      val eq_dec : X.t -> X.t -> sumbool

      val lt_dec : X.t -> X.t -> sumbool

      val eqb : X.t -> X.t -> bool
     end
   end
 end

module Make :
 functor (X:OrderedType) ->
 sig
  module Raw :
   sig
    module MX :
     sig
      module OrderTac :
       sig
        module OTF :
         sig
          type t = X.t

          val compare : t -> t -> comparison

          val eq_dec : t -> t -> sumbool
         end

        module TO :
         sig
          type t = OTF.t

          val compare : t -> t -> comparison

          val eq_dec : t -> t -> sumbool
         end
       end

      val eq_dec : X.t -> X.t -> sumbool

      val lt_dec : X.t -> X.t -> sumbool

      val eqb : X.t -> X.t -> bool
     end

    module ML :
     sig
     end

    type elt = X.t

    type t = elt list

    val empty : t

    val is_empty : t -> bool

    val mem : X.t -> X.t list -> bool

    val add : X.t -> X.t list -> X.t list

    val singleton : elt -> elt list

    val remove : X.t -> X.t list -> t

    val union : t -> t -> t

    val inter : t -> t -> t

    val diff : t -> t -> t

    val equal : t -> t -> bool

    val subset : X.t list -> X.t list -> bool

    val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

    val filter : (elt -> bool) -> t -> t

    val for_all : (elt -> bool) -> t -> bool

    val exists_ : (elt -> bool) -> t -> bool

    val partition : (elt -> bool) -> t -> (t, t) prod

    val cardinal : t -> nat

    val elements : t -> elt list

    val min_elt : t -> elt option

    val max_elt : t -> elt option

    val choose : t -> elt option

    val compare : X.t list -> X.t list -> comparison

    val inf : X.t -> X.t list -> bool

    val isok : X.t list -> bool

    module L :
     sig
      module MO :
       sig
        module OrderTac :
         sig
          module OTF :
           sig
            type t = X.t

            val compare : t -> t -> comparison

            val eq_dec : t -> t -> sumbool
           end

          module TO :
           sig
            type t = OTF.t

            val compare : t -> t -> comparison

            val eq_dec : t -> t -> sumbool
           end
         end

        val eq_dec : X.t -> X.t -> sumbool

        val lt_dec : X.t -> X.t -> sumbool

        val eqb : X.t -> X.t -> bool
       end
     end
   end

  module E :
   sig
    type t = X.t

    val compare : t -> t -> comparison

    val eq_dec : t -> t -> sumbool
   end

  type elt = X.t

  type t_ = Raw.t
    (* singleton inductive, whose constructor was Mkt *)

  val this : t_ -> Raw.t

  type t = t_

  val mem : elt -> t -> bool

  val add : elt -> t -> t

  val remove : elt -> t -> t

  val singleton : elt -> t

  val union : t -> t -> t

  val inter : t -> t -> t

  val diff : t -> t -> t

  val equal : t -> t -> bool

  val subset : t -> t -> bool

  val empty : t

  val is_empty : t -> bool

  val elements : t -> elt list

  val choose : t -> elt option

  val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

  val cardinal : t -> nat

  val filter : (elt -> bool) -> t -> t

  val for_all : (elt -> bool) -> t -> bool

  val exists_ : (elt -> bool) -> t -> bool

  val partition : (elt -> bool) -> t -> (t, t) prod

  val eq_dec : t -> t -> sumbool

  val compare : t -> t -> comparison

  val min_elt : t -> elt option

  val max_elt : t -> elt option
 end

module type OrderedTypeOrig =
 Coq_OrderedType

module Update_OT :
 functor (O:OrderedTypeOrig) ->
 sig
  type t = O.t

  val eq_dec : t -> t -> sumbool

  val compare : O.t -> O.t -> comparison
 end

module Coq_Make :
 functor (X:Coq_OrderedType) ->
 sig
  module X' :
   sig
    type t = X.t

    val eq_dec : t -> t -> sumbool

    val compare : X.t -> X.t -> comparison
   end

  module MSet :
   sig
    module Raw :
     sig
      module MX :
       sig
        module OrderTac :
         sig
          module OTF :
           sig
            type t = X.t

            val compare : X.t -> X.t -> comparison

            val eq_dec : X.t -> X.t -> sumbool
           end

          module TO :
           sig
            type t = X.t

            val compare : X.t -> X.t -> comparison

            val eq_dec : X.t -> X.t -> sumbool
           end
         end

        val eq_dec : X.t -> X.t -> sumbool

        val lt_dec : X.t -> X.t -> sumbool

        val eqb : X.t -> X.t -> bool
       end

      module ML :
       sig
       end

      type elt = X.t

      type t = elt list

      val empty : t

      val is_empty : t -> bool

      val mem : X.t -> X.t list -> bool

      val add : X.t -> X.t list -> X.t list

      val singleton : elt -> elt list

      val remove : X.t -> X.t list -> t

      val union : t -> t -> t

      val inter : t -> t -> t

      val diff : t -> t -> t

      val equal : t -> t -> bool

      val subset : X.t list -> X.t list -> bool

      val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

      val filter : (elt -> bool) -> t -> t

      val for_all : (elt -> bool) -> t -> bool

      val exists_ : (elt -> bool) -> t -> bool

      val partition : (elt -> bool) -> t -> (t, t) prod

      val cardinal : t -> nat

      val elements : t -> elt list

      val min_elt : t -> elt option

      val max_elt : t -> elt option

      val choose : t -> elt option

      val compare : X.t list -> X.t list -> comparison

      val inf : X.t -> X.t list -> bool

      val isok : X.t list -> bool

      module L :
       sig
        module MO :
         sig
          module OrderTac :
           sig
            module OTF :
             sig
              type t = X.t

              val compare : X.t -> X.t -> comparison

              val eq_dec : X.t -> X.t -> sumbool
             end

            module TO :
             sig
              type t = X.t

              val compare : X.t -> X.t -> comparison

              val eq_dec : X.t -> X.t -> sumbool
             end
           end

          val eq_dec : X.t -> X.t -> sumbool

          val lt_dec : X.t -> X.t -> sumbool

          val eqb : X.t -> X.t -> bool
         end
       end
     end

    module E :
     sig
      type t = X.t

      val compare : X.t -> X.t -> comparison

      val eq_dec : X.t -> X.t -> sumbool
     end

    type elt = X.t

    type t_ = Raw.t
      (* singleton inductive, whose constructor was Mkt *)

    val this : t_ -> Raw.t

    type t = t_

    val mem : elt -> t -> bool

    val add : elt -> t -> t

    val remove : elt -> t -> t

    val singleton : elt -> t

    val union : t -> t -> t

    val inter : t -> t -> t

    val diff : t -> t -> t

    val equal : t -> t -> bool

    val subset : t -> t -> bool

    val empty : t

    val is_empty : t -> bool

    val elements : t -> elt list

    val choose : t -> elt option

    val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

    val cardinal : t -> nat

    val filter : (elt -> bool) -> t -> t

    val for_all : (elt -> bool) -> t -> bool

    val exists_ : (elt -> bool) -> t -> bool

    val partition : (elt -> bool) -> t -> (t, t) prod

    val eq_dec : t -> t -> sumbool

    val compare : t -> t -> comparison

    val min_elt : t -> elt option

    val max_elt : t -> elt option
   end

  type elt = X.t

  type t = MSet.t

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

  module MF :
   sig
    val eqb : X.t -> X.t -> bool
   end

  val min_elt : t -> elt option

  val max_elt : t -> elt option

  val compare : t -> t -> t compare0

  module E :
   sig
    type t = X.t

    val compare : t -> t -> t compare0

    val eq_dec : t -> t -> sumbool
   end
 end

module WDecide_fun :
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
 sig
  module F :
   sig
    val eqb : E.t -> E.t -> bool
   end

  module FSetLogicalFacts :
   sig
   end

  module FSetDecideAuxiliary :
   sig
   end

  module FSetDecideTestCases :
   sig
   end
 end

module WProperties_fun :
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
 sig
  module Dec :
   sig
    module F :
     sig
      val eqb : E.t -> E.t -> bool
     end

    module FSetLogicalFacts :
     sig
     end

    module FSetDecideAuxiliary :
     sig
     end

    module FSetDecideTestCases :
     sig
     end
   end

  module FM :
   sig
    val eqb : E.t -> E.t -> bool
   end

  val coq_In_dec : M.elt -> M.t -> sumbool

  val of_list : M.elt list -> M.t

  val to_list : M.t -> M.elt list

  val fold_rec :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> __ -> 'a2) -> (M.elt ->
    'a1 -> M.t -> M.t -> __ -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_bis :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> M.t -> 'a1 -> __ -> 'a2 ->
    'a2) -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_nodep :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> 'a2 -> (M.elt -> 'a1 -> __ -> 'a2
    -> 'a2) -> 'a2

  val fold_rec_weak :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> (M.t -> M.t -> 'a1 -> __ -> 'a2 -> 'a2)
    -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> 'a2 -> 'a2) -> M.t -> 'a2

  val fold_rel :
    (M.elt -> 'a1 -> 'a1) -> (M.elt -> 'a2 -> 'a2) -> 'a1 -> 'a2 -> M.t ->
    'a3 -> (M.elt -> 'a1 -> 'a2 -> __ -> 'a3 -> 'a3) -> 'a3

  val set_induction :
    (M.t -> __ -> 'a1) -> (M.t -> M.t -> 'a1 -> M.elt -> __ -> __ -> 'a1) ->
    M.t -> 'a1

  val set_induction_bis :
    (M.t -> M.t -> __ -> 'a1 -> 'a1) -> 'a1 -> (M.elt -> M.t -> __ -> 'a1 ->
    'a1) -> M.t -> 'a1

  val cardinal_inv_2 : M.t -> nat -> M.elt

  val cardinal_inv_2b : M.t -> M.elt
 end

module WProperties :
 functor (M:WS) ->
 sig
  module Dec :
   sig
    module F :
     sig
      val eqb : M.E.t -> M.E.t -> bool
     end

    module FSetLogicalFacts :
     sig
     end

    module FSetDecideAuxiliary :
     sig
     end

    module FSetDecideTestCases :
     sig
     end
   end

  module FM :
   sig
    val eqb : M.E.t -> M.E.t -> bool
   end

  val coq_In_dec : M.elt -> M.t -> sumbool

  val of_list : M.elt list -> M.t

  val to_list : M.t -> M.elt list

  val fold_rec :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> __ -> 'a2) -> (M.elt ->
    'a1 -> M.t -> M.t -> __ -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_bis :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> M.t -> 'a1 -> __ -> 'a2 ->
    'a2) -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_nodep :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> 'a2 -> (M.elt -> 'a1 -> __ -> 'a2
    -> 'a2) -> 'a2

  val fold_rec_weak :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> (M.t -> M.t -> 'a1 -> __ -> 'a2 -> 'a2)
    -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> 'a2 -> 'a2) -> M.t -> 'a2

  val fold_rel :
    (M.elt -> 'a1 -> 'a1) -> (M.elt -> 'a2 -> 'a2) -> 'a1 -> 'a2 -> M.t ->
    'a3 -> (M.elt -> 'a1 -> 'a2 -> __ -> 'a3 -> 'a3) -> 'a3

  val set_induction :
    (M.t -> __ -> 'a1) -> (M.t -> M.t -> 'a1 -> M.elt -> __ -> __ -> 'a1) ->
    M.t -> 'a1

  val set_induction_bis :
    (M.t -> M.t -> __ -> 'a1 -> 'a1) -> 'a1 -> (M.elt -> M.t -> __ -> 'a1 ->
    'a1) -> M.t -> 'a1

  val cardinal_inv_2 : M.t -> nat -> M.elt

  val cardinal_inv_2b : M.t -> M.elt
 end

module Properties :
 functor (M:WS) ->
 sig
  module Dec :
   sig
    module F :
     sig
      val eqb : M.E.t -> M.E.t -> bool
     end

    module FSetLogicalFacts :
     sig
     end

    module FSetDecideAuxiliary :
     sig
     end

    module FSetDecideTestCases :
     sig
     end
   end

  module FM :
   sig
    val eqb : M.E.t -> M.E.t -> bool
   end

  val coq_In_dec : M.elt -> M.t -> sumbool

  val of_list : M.elt list -> M.t

  val to_list : M.t -> M.elt list

  val fold_rec :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> __ -> 'a2) -> (M.elt ->
    'a1 -> M.t -> M.t -> __ -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_bis :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> (M.t -> M.t -> 'a1 -> __ -> 'a2 ->
    'a2) -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_nodep :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> M.t -> 'a2 -> (M.elt -> 'a1 -> __ -> 'a2
    -> 'a2) -> 'a2

  val fold_rec_weak :
    (M.elt -> 'a1 -> 'a1) -> 'a1 -> (M.t -> M.t -> 'a1 -> __ -> 'a2 -> 'a2)
    -> 'a2 -> (M.elt -> 'a1 -> M.t -> __ -> 'a2 -> 'a2) -> M.t -> 'a2

  val fold_rel :
    (M.elt -> 'a1 -> 'a1) -> (M.elt -> 'a2 -> 'a2) -> 'a1 -> 'a2 -> M.t ->
    'a3 -> (M.elt -> 'a1 -> 'a2 -> __ -> 'a3 -> 'a3) -> 'a3

  val set_induction :
    (M.t -> __ -> 'a1) -> (M.t -> M.t -> 'a1 -> M.elt -> __ -> __ -> 'a1) ->
    M.t -> 'a1

  val set_induction_bis :
    (M.t -> M.t -> __ -> 'a1 -> 'a1) -> 'a1 -> (M.elt -> M.t -> __ -> 'a1 ->
    'a1) -> M.t -> 'a1

  val cardinal_inv_2 : M.t -> nat -> M.elt

  val cardinal_inv_2b : M.t -> M.elt
 end

module NatSet :
 sig
  module X' :
   sig
    type t = nat

    val eq_dec : nat -> nat -> sumbool

    val compare : nat -> nat -> comparison
   end

  module MSet :
   sig
    module Raw :
     sig
      module MX :
       sig
        module OrderTac :
         sig
          module OTF :
           sig
            type t = nat

            val compare : nat -> nat -> comparison

            val eq_dec : nat -> nat -> sumbool
           end

          module TO :
           sig
            type t = nat

            val compare : nat -> nat -> comparison

            val eq_dec : nat -> nat -> sumbool
           end
         end

        val eq_dec : nat -> nat -> sumbool

        val lt_dec : nat -> nat -> sumbool

        val eqb : nat -> nat -> bool
       end

      module ML :
       sig
       end

      type elt = nat

      type t = elt list

      val empty : t

      val is_empty : t -> bool

      val mem : nat -> nat list -> bool

      val add : nat -> nat list -> nat list

      val singleton : elt -> elt list

      val remove : nat -> nat list -> t

      val union : t -> t -> t

      val inter : t -> t -> t

      val diff : t -> t -> t

      val equal : t -> t -> bool

      val subset : nat list -> nat list -> bool

      val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

      val filter : (elt -> bool) -> t -> t

      val for_all : (elt -> bool) -> t -> bool

      val exists_ : (elt -> bool) -> t -> bool

      val partition : (elt -> bool) -> t -> (t, t) prod

      val cardinal : t -> nat

      val elements : t -> elt list

      val min_elt : t -> elt option

      val max_elt : t -> elt option

      val choose : t -> elt option

      val compare : nat list -> nat list -> comparison

      val inf : nat -> nat list -> bool

      val isok : nat list -> bool

      module L :
       sig
        module MO :
         sig
          module OrderTac :
           sig
            module OTF :
             sig
              type t = nat

              val compare : nat -> nat -> comparison

              val eq_dec : nat -> nat -> sumbool
             end

            module TO :
             sig
              type t = nat

              val compare : nat -> nat -> comparison

              val eq_dec : nat -> nat -> sumbool
             end
           end

          val eq_dec : nat -> nat -> sumbool

          val lt_dec : nat -> nat -> sumbool

          val eqb : nat -> nat -> bool
         end
       end
     end

    module E :
     sig
      type t = nat

      val compare : nat -> nat -> comparison

      val eq_dec : nat -> nat -> sumbool
     end

    type elt = nat

    type t_ = Raw.t
      (* singleton inductive, whose constructor was Mkt *)

    val this : t_ -> Raw.t

    type t = t_

    val mem : elt -> t -> bool

    val add : elt -> t -> t

    val remove : elt -> t -> t

    val singleton : elt -> t

    val union : t -> t -> t

    val inter : t -> t -> t

    val diff : t -> t -> t

    val equal : t -> t -> bool

    val subset : t -> t -> bool

    val empty : t

    val is_empty : t -> bool

    val elements : t -> elt list

    val choose : t -> elt option

    val fold : (elt -> 'a1 -> 'a1) -> t -> 'a1 -> 'a1

    val cardinal : t -> nat

    val filter : (elt -> bool) -> t -> t

    val for_all : (elt -> bool) -> t -> bool

    val exists_ : (elt -> bool) -> t -> bool

    val partition : (elt -> bool) -> t -> (t, t) prod

    val eq_dec : t -> t -> sumbool

    val compare : t -> t -> comparison

    val min_elt : t -> elt option

    val max_elt : t -> elt option
   end

  type elt = nat

  type t = MSet.t

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

  module MF :
   sig
    val eqb : nat -> nat -> bool
   end

  val min_elt : t -> elt option

  val max_elt : t -> elt option

  val compare : t -> t -> t compare0

  module E :
   sig
    type t = nat

    val compare : nat -> nat -> nat compare0

    val eq_dec : nat -> nat -> sumbool
   end
 end

module GeneralProperties :
 sig
  module Dec :
   sig
    module F :
     sig
      val eqb : nat -> nat -> bool
     end

    module FSetLogicalFacts :
     sig
     end

    module FSetDecideAuxiliary :
     sig
     end

    module FSetDecideTestCases :
     sig
     end
   end

  module FM :
   sig
    val eqb : nat -> nat -> bool
   end

  val coq_In_dec : NatSet.elt -> NatSet.t -> sumbool

  val of_list : NatSet.elt list -> NatSet.t

  val to_list : NatSet.t -> NatSet.elt list

  val fold_rec :
    (NatSet.elt -> 'a1 -> 'a1) -> 'a1 -> NatSet.t -> (NatSet.t -> __ -> 'a2)
    -> (NatSet.elt -> 'a1 -> NatSet.t -> NatSet.t -> __ -> __ -> __ -> 'a2 ->
    'a2) -> 'a2

  val fold_rec_bis :
    (NatSet.elt -> 'a1 -> 'a1) -> 'a1 -> NatSet.t -> (NatSet.t -> NatSet.t ->
    'a1 -> __ -> 'a2 -> 'a2) -> 'a2 -> (NatSet.elt -> 'a1 -> NatSet.t -> __
    -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_nodep :
    (NatSet.elt -> 'a1 -> 'a1) -> 'a1 -> NatSet.t -> 'a2 -> (NatSet.elt ->
    'a1 -> __ -> 'a2 -> 'a2) -> 'a2

  val fold_rec_weak :
    (NatSet.elt -> 'a1 -> 'a1) -> 'a1 -> (NatSet.t -> NatSet.t -> 'a1 -> __
    -> 'a2 -> 'a2) -> 'a2 -> (NatSet.elt -> 'a1 -> NatSet.t -> __ -> 'a2 ->
    'a2) -> NatSet.t -> 'a2

  val fold_rel :
    (NatSet.elt -> 'a1 -> 'a1) -> (NatSet.elt -> 'a2 -> 'a2) -> 'a1 -> 'a2 ->
    NatSet.t -> 'a3 -> (NatSet.elt -> 'a1 -> 'a2 -> __ -> 'a3 -> 'a3) -> 'a3

  val set_induction :
    (NatSet.t -> __ -> 'a1) -> (NatSet.t -> NatSet.t -> 'a1 -> NatSet.elt ->
    __ -> __ -> 'a1) -> NatSet.t -> 'a1

  val set_induction_bis :
    (NatSet.t -> NatSet.t -> __ -> 'a1 -> 'a1) -> 'a1 -> (NatSet.elt ->
    NatSet.t -> __ -> 'a1 -> 'a1) -> NatSet.t -> 'a1

  val cardinal_inv_2 : NatSet.t -> nat -> NatSet.elt

  val cardinal_inv_2b : NatSet.t -> NatSet.elt
 end

val extendible_by_one : NatSet.t -> nat -> NatSet.t -> NatSet.elt

val extendible_to_n : NatSet.t -> nat -> NatSet.t -> NatSet.t

val inductive_invariant :
  NatSet.t -> nat -> (NatSet.t -> __ -> __ -> NatSet.elt) -> nat -> NatSet.t

val aMM11262 :
  NatSet.t -> nat -> (NatSet.t -> __ -> __ -> NatSet.elt) -> NatSet.elt

val town_2 : NatSet.t

val subsets_2 :
  NatSet.t -> sumbool sumor sumor sumor sumor sumor sumor sumor sumor

val acquintance_2 : NatSet.t -> NatSet.elt

val social_citizen_2 : NatSet.elt
