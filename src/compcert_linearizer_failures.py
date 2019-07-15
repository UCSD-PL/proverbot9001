##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

compcert_failures = (
  [
      ["./driver/Complements.v", "Theorem transf_c_program_preservation"],
      ["./driver/Complements.v", "Theorem transf_c_program_is_refinement"],
      ["./backend/SelectDivProof.v", "Theorem eval_divuimm"],
      ["./backend/Stackingproof.v", "Lemma set_location"], # fails because of linearizing "eapply Mem.load_store_similar_2; eauto."
      ["./cfrontend/Cminorgenproof.v", "Lemma padding_freeable_invariant"],
      ["./cfrontend/Cminorgenproof.v", "Lemma match_callstack_alloc_right"],
      ["./backend/ValueDomain.v", "Lemma pincl_sound"],
      ["./backend/Unusedglobproof.v", "Lemma add_ref_globvar_incl"],
      ["./backend/Unusedglobproof.v", "Lemma initial_workset_incl"],
      ["./backend/Unusedglobproof.v", "Theorem used_globals_sound"],
      ["./backend/Unusedglobproof.v", "Theorem used_globals_incl"],
      ["./backend/Unusedglobproof.v", "Lemma match_stacks_incr"],
      ["./backend/Unusedglobproof.v", "Remark link_def_either"],

      ["./lib/Maps.v", "Remark xelements_empty"],
  ]
)
