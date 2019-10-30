auto.vo auto.glob auto.v.beautified: auto.v bases.vo defs.vo empty_test.vo inter.vo lattice_fixpoint.vo pl_path.vo refcorrect.vo semantics.vo signature.vo states_kill_empty.vo union.vo union_correct.vo inter_correct.vo states_kill_correct.vo coacc_test.vo non_coacc_kill.vo non_coacc_kill_correct.vo
auto.vio: auto.v bases.vio defs.vio empty_test.vio inter.vio lattice_fixpoint.vio pl_path.vio refcorrect.vio semantics.vio signature.vio states_kill_empty.vio union.vio union_correct.vio inter_correct.vio states_kill_correct.vio coacc_test.vio non_coacc_kill.vio non_coacc_kill_correct.vio
bases.vo bases.glob bases.v.beautified: bases.v
bases.vio: bases.v
coacc_test.vo coacc_test.glob coacc_test.v.beautified: coacc_test.v bases.vo defs.vo semantics.vo refcorrect.vo lattice_fixpoint.vo
coacc_test.vio: coacc_test.v bases.vio defs.vio semantics.vio refcorrect.vio lattice_fixpoint.vio
defs.vo defs.glob defs.v.beautified: defs.v bases.vo
defs.vio: defs.v bases.vio
empty_test.vo empty_test.glob empty_test.v.beautified: empty_test.v lattice_fixpoint.vo bases.vo defs.vo semantics.vo pl_path.vo
empty_test.vio: empty_test.v lattice_fixpoint.vio bases.vio defs.vio semantics.vio pl_path.vio
inter_correct.vo inter_correct.glob inter_correct.v.beautified: inter_correct.v bases.vo defs.vo semantics.vo signature.vo refcorrect.vo inter.vo
inter_correct.vio: inter_correct.v bases.vio defs.vio semantics.vio signature.vio refcorrect.vio inter.vio
inter.vo inter.glob inter.v.beautified: inter.v bases.vo defs.vo semantics.vo pl_path.vo signature.vo
inter.vio: inter.v bases.vio defs.vio semantics.vio pl_path.vio signature.vio
lattice_fixpoint.vo lattice_fixpoint.glob lattice_fixpoint.v.beautified: lattice_fixpoint.v bases.vo
lattice_fixpoint.vio: lattice_fixpoint.v bases.vio
non_coacc_kill_correct.vo non_coacc_kill_correct.glob non_coacc_kill_correct.v.beautified: non_coacc_kill_correct.v bases.vo defs.vo semantics.vo refcorrect.vo signature.vo lattice_fixpoint.vo coacc_test.vo non_coacc_kill.vo
non_coacc_kill_correct.vio: non_coacc_kill_correct.v bases.vio defs.vio semantics.vio refcorrect.vio signature.vio lattice_fixpoint.vio coacc_test.vio non_coacc_kill.vio
non_coacc_kill.vo non_coacc_kill.glob non_coacc_kill.v.beautified: non_coacc_kill.v bases.vo defs.vo semantics.vo refcorrect.vo lattice_fixpoint.vo coacc_test.vo
non_coacc_kill.vio: non_coacc_kill.v bases.vio defs.vio semantics.vio refcorrect.vio lattice_fixpoint.vio coacc_test.vio
pl_path.vo pl_path.glob pl_path.v.beautified: pl_path.v bases.vo defs.vo semantics.vo signature.vo
pl_path.vio: pl_path.v bases.vio defs.vio semantics.vio signature.vio
refcorrect.vo refcorrect.glob refcorrect.v.beautified: refcorrect.v bases.vo defs.vo
refcorrect.vio: refcorrect.v bases.vio defs.vio
semantics.vo semantics.glob semantics.v.beautified: semantics.v bases.vo defs.vo
semantics.vio: semantics.v bases.vio defs.vio
signature.vo signature.glob signature.v.beautified: signature.v bases.vo defs.vo semantics.vo
signature.vio: signature.v bases.vio defs.vio semantics.vio
states_kill_correct.vo states_kill_correct.glob states_kill_correct.v.beautified: states_kill_correct.v bases.vo defs.vo semantics.vo signature.vo pl_path.vo refcorrect.vo states_kill_empty.vo lattice_fixpoint.vo empty_test.vo
states_kill_correct.vio: states_kill_correct.v bases.vio defs.vio semantics.vio signature.vio pl_path.vio refcorrect.vio states_kill_empty.vio lattice_fixpoint.vio empty_test.vio
states_kill_empty.vo states_kill_empty.glob states_kill_empty.v.beautified: states_kill_empty.v bases.vo defs.vo semantics.vo lattice_fixpoint.vo signature.vo pl_path.vo empty_test.vo
states_kill_empty.vio: states_kill_empty.v bases.vio defs.vio semantics.vio lattice_fixpoint.vio signature.vio pl_path.vio empty_test.vio
union_correct.vo union_correct.glob union_correct.v.beautified: union_correct.v bases.vo defs.vo semantics.vo signature.vo refcorrect.vo union.vo
union_correct.vio: union_correct.v bases.vio defs.vio semantics.vio signature.vio refcorrect.vio union.vio
union.vo union.glob union.v.beautified: union.v bases.vo defs.vo semantics.vo signature.vo refcorrect.vo
union.vio: union.v bases.vio defs.vio semantics.vio signature.vio refcorrect.vio
