beta.vo beta.glob beta.v.beautified: beta.v tactics.vo imports.vo ett_inf.vo ett_ott.vo ett_ind.vo ext_context_fv.vo ext_wf.vo fc_wf.vo utils.vo erase_syntax.vo toplevel.vo ett_value.vo
beta.vio: beta.v tactics.vio imports.vio ett_inf.vio ett_ott.vio ett_ind.vio ext_context_fv.vio ext_wf.vio fc_wf.vio utils.vio erase_syntax.vio toplevel.vio ett_value.vio
congruence.vo congruence.glob congruence.v.beautified: congruence.v sigs.vo imports.vo tactics.vo ett_ott.vo ett_inf.vo ett_inf_cs.vo ett_ind.vo ett_par.vo utils.vo fc_unique.vo fc_invert.vo erase.vo
congruence.vio: congruence.v sigs.vio imports.vio tactics.vio ett_ott.vio ett_inf.vio ett_inf_cs.vio ett_ind.vio ett_par.vio utils.vio fc_unique.vio fc_invert.vio erase.vio
dep_prog.vo dep_prog.glob dep_prog.v.beautified: dep_prog.v imports.vo
dep_prog.vio: dep_prog.v imports.vio
erase_syntax.vo erase_syntax.glob erase_syntax.v.beautified: erase_syntax.v ett_inf_cs.vo ett_ind.vo imports.vo tactics.vo
erase_syntax.vio: erase_syntax.v ett_inf_cs.vio ett_ind.vio imports.vio tactics.vio
erase.vo erase.glob erase.v.beautified: erase.v sigs.vo utils.vo ett_inf_cs.vo ett_ind.vo imports.vo tactics.vo erase_syntax.vo ext_red.vo fc_invert.vo fc_unique.vo ett_par.vo toplevel.vo fc_context_fv.vo
erase.vio: erase.v sigs.vio utils.vio ett_inf_cs.vio ett_ind.vio imports.vio tactics.vio erase_syntax.vio ext_red.vio fc_invert.vio fc_unique.vio ett_par.vio toplevel.vio fc_context_fv.vio
ett_ind.vo ett_ind.glob ett_ind.v.beautified: ett_ind.v utils.vo imports.vo fset_facts.vo ett_inf.vo tactics.vo
ett_ind.vio: ett_ind.v utils.vio imports.vio fset_facts.vio ett_inf.vio tactics.vio
ett_inf_cs.vo ett_inf_cs.glob ett_inf_cs.v.beautified: ett_inf_cs.v ett_inf.vo imports.vo
ett_inf_cs.vio: ett_inf_cs.v ett_inf.vio imports.vio
ett_inf.vo ett_inf.glob ett_inf.v.beautified: ett_inf.v ett_ott.vo
ett_inf.vio: ett_inf.v ett_ott.vio
ett_ott.vo ett_ott.glob ett_ott.v.beautified: ett_ott.v
ett_ott.vio: ett_ott.v
ett_par.vo ett_par.glob ett_par.v.beautified: ett_par.v tactics.vo imports.vo utils.vo ett_inf.vo ett_ott.vo ett_ind.vo ext_context_fv.vo ext_wf.vo erase_syntax.vo toplevel.vo ett_value.vo
ett_par.vio: ett_par.v tactics.vio imports.vio utils.vio ett_inf.vio ett_ott.vio ett_ind.vio ext_context_fv.vio ext_wf.vio erase_syntax.vio toplevel.vio ett_value.vio
ett_value.vo ett_value.glob ett_value.v.beautified: ett_value.v tactics.vo imports.vo ett_inf.vo ett_ott.vo ett_ind.vo ext_context_fv.vo ext_wf.vo utils.vo erase_syntax.vo toplevel.vo
ett_value.vio: ett_value.v tactics.vio imports.vio ett_inf.vio ett_ott.vio ett_ind.vio ext_context_fv.vio ext_wf.vio utils.vio erase_syntax.vio toplevel.vio
ext_consist.vo ext_consist.glob ext_consist.v.beautified: ext_consist.v sigs.vo imports.vo utils.vo tactics.vo ett_ott.vo ett_inf.vo ett_ind.vo ett_par.vo erase_syntax.vo ext_red_one.vo ext_red.vo ext_wf.vo
ext_consist.vio: ext_consist.v sigs.vio imports.vio utils.vio tactics.vio ett_ott.vio ett_inf.vio ett_ind.vio ett_par.vio erase_syntax.vio ext_red_one.vio ext_red.vio ext_wf.vio
ext_context_fv.vo ext_context_fv.glob ext_context_fv.v.beautified: ext_context_fv.v tactics.vo ett_inf.vo utils.vo imports.vo ett_ind.vo toplevel.vo
ext_context_fv.vio: ext_context_fv.v tactics.vio ett_inf.vio utils.vio imports.vio ett_ind.vio toplevel.vio
ext_invert.vo ext_invert.glob ext_invert.v.beautified: ext_invert.v sigs.vo imports.vo ett_ott.vo ett_inf.vo ett_ind.vo ett_par.vo ext_wf.vo utils.vo
ext_invert.vio: ext_invert.v sigs.vio imports.vio ett_ott.vio ett_inf.vio ett_ind.vio ett_par.vio ext_wf.vio utils.vio
ext_red_one.vo ext_red_one.glob ext_red_one.v.beautified: ext_red_one.v sigs.vo imports.vo utils.vo tactics.vo ett_ott.vo ett_inf.vo ett_ind.vo ett_par.vo erase_syntax.vo
ext_red_one.vio: ext_red_one.v sigs.vio imports.vio utils.vio tactics.vio ett_ott.vio ett_inf.vio ett_ind.vio ett_par.vio erase_syntax.vio
ext_red.vo ext_red.glob ext_red.v.beautified: ext_red.v sigs.vo imports.vo ett_ott.vo ett_inf.vo ett_par.vo ett_ind.vo ext_wf.vo ext_red_one.vo tactics.vo
ext_red.vio: ext_red.v sigs.vio imports.vio ett_ott.vio ett_inf.vio ett_par.vio ett_ind.vio ext_wf.vio ext_red_one.vio tactics.vio
ext_subst.vo ext_subst.glob ext_subst.v.beautified: ext_subst.v sigs.vo tactics.vo imports.vo utils.vo fset_facts.vo ett_inf.vo ett_ind.vo beta.vo ext_wf.vo ett_value.vo
ext_subst.vio: ext_subst.v sigs.vio tactics.vio imports.vio utils.vio fset_facts.vio ett_inf.vio ett_ind.vio beta.vio ext_wf.vio ett_value.vio
ext_weak.vo ext_weak.glob ext_weak.v.beautified: ext_weak.v sigs.vo tactics.vo utils.vo imports.vo ett_inf.vo ett_par.vo ett_ind.vo
ext_weak.vio: ext_weak.v sigs.vio tactics.vio utils.vio imports.vio ett_inf.vio ett_par.vio ett_ind.vio
ext_wf.vo ext_wf.glob ext_wf.v.beautified: ext_wf.v imports.vo ett_inf.vo ett_ind.vo tactics.vo utils.vo sigs.vo toplevel.vo
ext_wf.vio: ext_wf.v imports.vio ett_inf.vio ett_ind.vio tactics.vio utils.vio sigs.vio toplevel.vio
fc_consist.vo fc_consist.glob fc_consist.v.beautified: fc_consist.v sigs.vo tactics.vo imports.vo ett_inf.vo ett_ott.vo ett_ind.vo ext_wf.vo ett_par.vo ett_inf_cs.vo erase_syntax.vo fc_invert.vo ext_consist.vo erase.vo fc_head_reduction.vo fc_preservation.vo ext_subst.vo
fc_consist.vio: fc_consist.v sigs.vio tactics.vio imports.vio ett_inf.vio ett_ott.vio ett_ind.vio ext_wf.vio ett_par.vio ett_inf_cs.vio erase_syntax.vio fc_invert.vio ext_consist.vio erase.vio fc_head_reduction.vio fc_preservation.vio ext_subst.vio
fc_context_fv.vo fc_context_fv.glob fc_context_fv.v.beautified: fc_context_fv.v tactics.vo ett_inf.vo utils.vo imports.vo ett_ind.vo
fc_context_fv.vio: fc_context_fv.v tactics.vio ett_inf.vio utils.vio imports.vio ett_ind.vio
fc_dec_aux.vo fc_dec_aux.glob fc_dec_aux.v.beautified: fc_dec_aux.v sigs.vo imports.vo dep_prog.vo ett_ind.vo ett_par.vo erase_syntax.vo
fc_dec_aux.vio: fc_dec_aux.v sigs.vio imports.vio dep_prog.vio ett_ind.vio ett_par.vio erase_syntax.vio
fc_dec_fuel.vo fc_dec_fuel.glob fc_dec_fuel.v.beautified: fc_dec_fuel.v sigs.vo imports.vo ett_ind.vo
fc_dec_fuel.vio: fc_dec_fuel.v sigs.vio imports.vio ett_ind.vio
fc_dec_fun.vo fc_dec_fun.glob fc_dec_fun.v.beautified: fc_dec_fun.v sigs.vo fc_dec_fuel.vo fc_dec_aux.vo imports.vo ett_inf_cs.vo ett_ind.vo fc_invert.vo dep_prog.vo toplevel.vo fc_get.vo fc_context_fv.vo
fc_dec_fun.vio: fc_dec_fun.v sigs.vio fc_dec_fuel.vio fc_dec_aux.vio imports.vio ett_inf_cs.vio ett_ind.vio fc_invert.vio dep_prog.vio toplevel.vio fc_get.vio fc_context_fv.vio
fc_dec.vo fc_dec.glob fc_dec.v.beautified: fc_dec.v sigs.vo fc_dec_fuel.vo fc_dec_fun.vo fc_dec_aux.vo imports.vo ett_ind.vo fc_invert.vo
fc_dec.vio: fc_dec.v sigs.vio fc_dec_fuel.vio fc_dec_fun.vio fc_dec_aux.vio imports.vio ett_ind.vio fc_invert.vio
fc_get.vo fc_get.glob fc_get.v.beautified: fc_get.v fc_invert.vo toplevel.vo erase_syntax.vo sigs.vo fc_unique.vo fc_context_fv.vo
fc_get.vio: fc_get.v fc_invert.vio toplevel.vio erase_syntax.vio sigs.vio fc_unique.vio fc_context_fv.vio
fc_head_reduction.vo fc_head_reduction.glob fc_head_reduction.v.beautified: fc_head_reduction.v sigs.vo fc_wf.vo imports.vo utils.vo tactics.vo ett_ott.vo ett_inf.vo ett_ind.vo ett_par.vo erase_syntax.vo ext_red_one.vo
fc_head_reduction.vio: fc_head_reduction.v sigs.vio fc_wf.vio imports.vio utils.vio tactics.vio ett_ott.vio ett_inf.vio ett_ind.vio ett_par.vio erase_syntax.vio ext_red_one.vio
fc_invert.vo fc_invert.glob fc_invert.v.beautified: fc_invert.v sigs.vo imports.vo tactics.vo fset_facts.vo ett_ott.vo ett_inf.vo ett_inf_cs.vo ett_ind.vo erase_syntax.vo fc_unique.vo fc_wf.vo toplevel.vo fc_context_fv.vo
fc_invert.vio: fc_invert.v sigs.vio imports.vio tactics.vio fset_facts.vio ett_ott.vio ett_inf.vio ett_inf_cs.vio ett_ind.vio erase_syntax.vio fc_unique.vio fc_wf.vio toplevel.vio fc_context_fv.vio
fc_preservation.vo fc_preservation.glob fc_preservation.v.beautified: fc_preservation.v sigs.vo imports.vo tactics.vo ett_ott.vo ett_inf.vo ett_inf_cs.vo ett_ind.vo ett_par.vo ext_invert.vo ext_red.vo ext_red_one.vo erase_syntax.vo fc_invert.vo fc_unique.vo
fc_preservation.vio: fc_preservation.v sigs.vio imports.vio tactics.vio ett_ott.vio ett_inf.vio ett_inf_cs.vio ett_ind.vio ett_par.vio ext_invert.vio ext_red.vio ext_red_one.vio erase_syntax.vio fc_invert.vio fc_unique.vio
fc_subst.vo fc_subst.glob fc_subst.v.beautified: fc_subst.v sigs.vo tactics.vo imports.vo utils.vo ett_inf.vo ett_ind.vo fset_facts.vo erase_syntax.vo ett_par.vo beta.vo fc_wf.vo fc_context_fv.vo
fc_subst.vio: fc_subst.v sigs.vio tactics.vio imports.vio utils.vio ett_inf.vio ett_ind.vio fset_facts.vio erase_syntax.vio ett_par.vio beta.vio fc_wf.vio fc_context_fv.vio
fc_unique.vo fc_unique.glob fc_unique.v.beautified: fc_unique.v sigs.vo ett_inf_cs.vo ett_ind.vo imports.vo tactics.vo ett_par.vo
fc_unique.vio: fc_unique.v sigs.vio ett_inf_cs.vio ett_ind.vio imports.vio tactics.vio ett_par.vio
fc_weak.vo fc_weak.glob fc_weak.v.beautified: fc_weak.v sigs.vo imports.vo ett_inf.vo ett_ind.vo tactics.vo ett_par.vo erase_syntax.vo fc_wf.vo
fc_weak.vio: fc_weak.v sigs.vio imports.vio ett_inf.vio ett_ind.vio tactics.vio ett_par.vio erase_syntax.vio fc_wf.vio
fc_wf.vo fc_wf.glob fc_wf.v.beautified: fc_wf.v sigs.vo imports.vo ett_inf.vo ett_ind.vo tactics.vo toplevel.vo
fc_wf.vio: fc_wf.v sigs.vio imports.vio ett_inf.vio ett_ind.vio tactics.vio toplevel.vio
fix_typing.vo fix_typing.glob fix_typing.v.beautified: fix_typing.v ett_ott.vo ett_inf.vo
fix_typing.vio: fix_typing.v ett_ott.vio ett_inf.vio
fset_facts.vo fset_facts.glob fset_facts.v.beautified: fset_facts.v tactics.vo ett_inf.vo imports.vo
fset_facts.vio: fset_facts.v tactics.vio ett_inf.vio imports.vio
imports.vo imports.glob imports.v.beautified: imports.v ett_ott.vo
imports.vio: imports.v ett_ott.vio
main.vo main.glob main.v.beautified: main.v ext_wf.vo ext_weak.vo ext_subst.vo ext_invert.vo ext_red.vo ext_red_one.vo fc_wf.vo fc_weak.vo fc_subst.vo fc_unique.vo fc_invert.vo fc_get.vo fc_dec.vo fc_preservation.vo fc_consist.vo fc_head_reduction.vo erase.vo ext_consist.vo
main.vio: main.v ext_wf.vio ext_weak.vio ext_subst.vio ext_invert.vio ext_red.vio ext_red_one.vio fc_wf.vio fc_weak.vio fc_subst.vio fc_unique.vio fc_invert.vio fc_get.vio fc_dec.vio fc_preservation.vio fc_consist.vio fc_head_reduction.vio erase.vio ext_consist.vio
sigs.vo sigs.glob sigs.v.beautified: sigs.v imports.vo ett_ott.vo utils.vo
sigs.vio: sigs.v imports.vio ett_ott.vio utils.vio
tactics.vo tactics.glob tactics.v.beautified: tactics.v imports.vo ett_inf.vo
tactics.vio: tactics.v imports.vio ett_inf.vio
toplevel.vo toplevel.glob toplevel.v.beautified: toplevel.v tactics.vo imports.vo ett_inf.vo ett_ott.vo ett_ind.vo utils.vo fix_typing.vo
toplevel.vio: toplevel.v tactics.vio imports.vio ett_inf.vio ett_ott.vio ett_ind.vio utils.vio fix_typing.vio
utils.vo utils.glob utils.v.beautified: utils.v imports.vo
utils.vio: utils.v imports.vio
