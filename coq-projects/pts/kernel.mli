
type __ = Obj.t

type nat =
| O
| S of nat

type ('a, 'b) sum =
| Inl of 'a
| Inr of 'b

type ('a, 'b) prod =
| Pair of 'a * 'b

type 'a list =
| Nil
| Cons of 'a * 'a list

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type 'a sig2 = 'a
  (* singleton inductive, whose constructor was exist2 *)

type 'a exc = 'a option

val value : 'a1 -> 'a1 option

val error : 'a1 option

val pred : nat -> nat

val add : nat -> nat -> nat

module Nat :
 sig
  val eq_dec : nat -> nat -> bool
 end

val lt_eq_lt_dec : nat -> nat -> bool option

val le_lt_dec : nat -> nat -> bool

val le_gt_dec : nat -> nat -> bool

val list_item : 'a1 list -> nat -> 'a1 option

val max_nat : nat -> nat -> nat

type decide = bool

type 'a ppal_dec = 'a option

type 'sort term =
| Srt of 'sort
| Ref of nat
| Abs of 'sort term * 'sort term
| App of 'sort term * 'sort term
| Prod of 'sort term * 'sort term

val lift_rec : nat -> 'a1 term -> nat -> 'a1 term

val lift : nat -> 'a1 term -> 'a1 term

val subst_rec : 'a1 term -> 'a1 term -> nat -> 'a1 term

val subst : 'a1 term -> 'a1 term -> 'a1 term

type 'sort decl =
| Ax of 'sort term
| Def of 'sort term * 'sort term

val typ_of_decl : 'a1 decl -> 'a1 term

type 'sort env = 'sort decl list

type 'sort basic_rule =
| Build_Basic_rule

type 'sort subtyping_rule =
  'sort basic_rule
  (* singleton inductive, whose constructor was Build_Subtyping_rule *)

type 'sort pTS_sub_spec =
  'sort subtyping_rule
  (* singleton inductive, whose constructor was Build_PTS_sub_spec *)

val is_a_sort : 'a1 term -> decide

type 'sort red_to_sort_dec = 'sort ppal_dec

type 'sort red_to_wf_prod_dec = ('sort term, 'sort term) prod option

type 'sort pTS_algos = { pa_lift : (nat -> 'sort term -> 'sort term);
                         pa_subst : ('sort term -> 'sort term -> 'sort term);
                         pa_infer_axiom : ('sort -> 'sort ppal_dec);
                         pa_least_sort : ('sort env -> 'sort term -> __ ->
                                         'sort red_to_sort_dec);
                         pa_infer_rule : ('sort -> 'sort -> 'sort);
                         pa_least_prod : ('sort env -> 'sort term -> __ ->
                                         'sort red_to_wf_prod_dec);
                         pa_le_type_dec : ('sort env -> 'sort term -> 'sort
                                          term -> __ -> __ -> decide) }

val pa_lift : 'a1 pTS_sub_spec -> 'a1 pTS_algos -> nat -> 'a1 term -> 'a1 term

val pa_subst :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 term -> 'a1 term -> 'a1 term

val pa_infer_axiom : 'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 -> 'a1 ppal_dec

val pa_least_sort :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1
  red_to_sort_dec

val pa_infer_rule : 'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 -> 'a1 -> 'a1

val pa_least_prod :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1
  red_to_wf_prod_dec

val pa_le_type_dec :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1 term ->
  decide

type 'sort type_error =
| Under of 'sort term * 'sort type_error
| Expected_type of 'sort term * 'sort term * 'sort term
| Topsort of 'sort
| Db_error of nat
| Lambda_topsort of 'sort term * 'sort
| Not_a_type of 'sort term * 'sort term
| Not_a_fun of 'sort term * 'sort term
| Apply_err of 'sort term * 'sort term * 'sort term * 'sort term

type 'sort infer_ppal_type = ('sort term, 'sort type_error sig2) sum

type 'sort wft_dec =
| Wft_ok
| Wft_fail of 'sort type_error

type 'sort check_dec =
| Chk_ok
| Chk_fail of 'sort type_error

type 'sort decl_dec =
| Dcl_ok
| Dcl_fail of 'sort type_error

type 'sort pTS_TC = { ptc_inf_ppal_type : ('sort env -> 'sort term -> __ ->
                                          'sort infer_ppal_type);
                      ptc_chk_typ : ('sort env -> 'sort term -> 'sort term ->
                                    __ -> 'sort check_dec);
                      ptc_add_typ : ('sort env -> 'sort term -> __ -> 'sort
                                    decl_dec);
                      ptc_add_def : ('sort env -> 'sort term -> 'sort term ->
                                    __ -> 'sort decl_dec);
                      ptc_chk_wk : ('sort env -> 'sort term -> 'sort term ->
                                   __ -> __ -> 'sort check_dec);
                      ptc_chk_wft : ('sort env -> 'sort term -> __ -> 'sort
                                    wft_dec) }

val ptc_inf_ppal_type :
  'a1 pTS_sub_spec -> 'a1 pTS_TC -> 'a1 env -> 'a1 term -> 'a1 infer_ppal_type

val ptc_chk_typ :
  'a1 pTS_sub_spec -> 'a1 pTS_TC -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1
  check_dec

val ptc_add_typ :
  'a1 pTS_sub_spec -> 'a1 pTS_TC -> 'a1 env -> 'a1 term -> 'a1 decl_dec

val ptc_add_def :
  'a1 pTS_sub_spec -> 'a1 pTS_TC -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1
  decl_dec

val ptc_chk_wft :
  'a1 pTS_sub_spec -> 'a1 pTS_TC -> 'a1 env -> 'a1 term -> 'a1 wft_dec

val not_topsort :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 term -> 'a1 sig2 option

val fix_chk_wk :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> ('a1 term -> 'a1 env -> __ -> 'a1
  infer_ppal_type) -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1 check_dec

val fix_add_typ :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> ('a1 term -> 'a1 env -> __ -> 'a1
  infer_ppal_type) -> 'a1 env -> 'a1 term -> 'a1 decl_dec

val infer_ref :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> nat -> 'a1 infer_ppal_type

val app_chk_err : 'a1 term -> 'a1 term -> 'a1 type_error -> 'a1 type_error

val infer_app :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> ('a1 term -> 'a1 env -> __ -> 'a1
  infer_ppal_type) -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1 infer_ppal_type

val infer_sort :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 -> 'a1 infer_ppal_type

val infer_abs :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> ('a1 term -> 'a1 env -> __ -> 'a1
  infer_ppal_type) -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1 infer_ppal_type

val infer_prod :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> ('a1 term -> 'a1 env -> __ -> 'a1
  infer_ppal_type) -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1 infer_ppal_type

val full_ppal_type :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 term -> 'a1 env -> 'a1
  infer_ppal_type

val tmp_add_typ :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1 decl_dec

val tmp_check_typ_when_wf :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1 term -> 'a1
  check_dec

val tmp_check_wf_type :
  'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 env -> 'a1 term -> 'a1 wft_dec

val full_type_checker : 'a1 pTS_sub_spec -> 'a1 pTS_algos -> 'a1 pTS_TC

val r_rt_basic_rule : 'a1 basic_rule -> 'a1 basic_rule

val union_basic_rule : 'a1 basic_rule -> 'a1 basic_rule -> 'a1 basic_rule

val canonical_subtyping : 'a1 basic_rule -> 'a1 subtyping_rule

val cR_WHNF_convert_hn :
  ('a1 -> 'a1 -> decide) -> 'a1 basic_rule -> ('a1 env -> 'a1 term -> __ ->
  'a1 term sig2) -> 'a1 env -> 'a1 term -> 'a1 term -> decide

type 'sort cTS_spec =
  'sort basic_rule
  (* singleton inductive, whose constructor was Build_CTS_spec *)

val cumul_rule : 'a1 cTS_spec -> 'a1 basic_rule

val cts_pts_functor : 'a1 cTS_spec -> 'a1 pTS_sub_spec

type 'sort subtype_dec_CTS = { scts_whnf : ('sort env -> 'sort term -> __ ->
                                           'sort term sig2);
                               scts_convert_hn : ('sort env -> 'sort term ->
                                                 'sort term -> __ -> __ ->
                                                 decide);
                               scts_rt_univ_dec : ('sort -> 'sort -> decide) }

val scts_whnf :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 env -> 'a1 term -> 'a1 term sig2

val scts_convert_hn :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 env -> 'a1 term -> 'a1 term ->
  decide

val scts_rt_univ_dec :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 -> 'a1 -> decide

val cR_WHNF_inv_cumul_dec :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 env -> 'a1 term -> 'a1 term ->
  decide

val cR_WHNF_cumul_dec :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 env -> 'a1 term -> 'a1 term ->
  decide

type 'sort norm_sound_CTS = { ncts_axiom : ('sort -> 'sort ppal_dec);
                              ncts_rule : ('sort -> 'sort -> 'sort sig2) }

val ncts_axiom : 'a1 cTS_spec -> 'a1 norm_sound_CTS -> 'a1 -> 'a1 ppal_dec

val ncts_rule : 'a1 cTS_spec -> 'a1 norm_sound_CTS -> 'a1 -> 'a1 -> 'a1 sig2

val cumul_least_sort :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 norm_sound_CTS -> 'a1 env -> 'a1
  term -> 'a1 red_to_sort_dec

val cumul_least_prod :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 norm_sound_CTS -> 'a1 env -> 'a1
  term -> 'a1 red_to_wf_prod_dec

val cumul_infer_axiom :
  'a1 cTS_spec -> 'a1 norm_sound_CTS -> 'a1 -> 'a1 ppal_dec

val cumul_infer_rule :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 norm_sound_CTS -> 'a1 -> 'a1 ->
  'a1

val cts_prim_algos :
  'a1 cTS_spec -> 'a1 subtype_dec_CTS -> 'a1 norm_sound_CTS -> 'a1 pTS_algos

val app_list : 'a1 term list -> 'a1 term -> 'a1 term

val beta_rule : 'a1 basic_rule

val delta_rule : 'a1 basic_rule

val delta_reduce : nat -> 'a1 env -> 'a1 term option

val beta_delta_rule : 'a1 basic_rule

val bd_whnf_rec : 'a1 env -> 'a1 term -> 'a1 term list -> 'a1 term sig2

val beta_delta_whnf : 'a1 env -> 'a1 term -> 'a1 term sig2

type gen_sort =
| Gprop
| Gset
| Gtype of nat
| Gtypeset of nat

type calc =
| Pos
| Neg

type srt_v6 =
| Sprop of calc
| Stype of nat

val calc_dec : calc -> calc -> decide

val v6_sort_dec : srt_v6 -> srt_v6 -> decide

val univ_v6_dec : srt_v6 -> srt_v6 -> decide

val v6_inf_axiom : srt_v6 -> srt_v6

val v6_inf_rule : srt_v6 -> srt_v6 -> srt_v6 sig2

val sort_of_gen : gen_sort -> srt_v6 exc

val gen_of_sort : srt_v6 -> gen_sort

type trm_v6 = srt_v6 term

type env_v6 = srt_v6 env

val v6 : srt_v6 cTS_spec

val v6_pts : srt_v6 pTS_sub_spec

val whnf : env_v6 -> trm_v6 -> trm_v6 sig2

val bd_conv_hnf : env_v6 -> trm_v6 -> trm_v6 -> decide

val v6_is_subtype_dec : srt_v6 subtype_dec_CTS

val v6_is_norm_sound : srt_v6 norm_sound_CTS

val infer_type : env_v6 -> trm_v6 -> srt_v6 infer_ppal_type

val check_wf_type : env_v6 -> trm_v6 -> srt_v6 wft_dec

val check_type : env_v6 -> trm_v6 -> trm_v6 -> srt_v6 check_dec

val add_type : env_v6 -> trm_v6 -> srt_v6 decl_dec

val add_def : env_v6 -> trm_v6 -> trm_v6 -> srt_v6 decl_dec
