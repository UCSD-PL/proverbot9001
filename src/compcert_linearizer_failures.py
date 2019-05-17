compcert_failures = (
  [
      ["./driver/Complements.v", "Theorem transf_c_program_preservation"],
      ["./driver/Complements.v", "Theorem transf_c_program_is_refinement"],
      ["./backend/SelectDivProof.v", "Theorem eval_divuimm"],
      ["./backend/Stackingproof.v", "Lemma set_location"], # fails because of linearizing "eapply Mem.load_store_similar_2; eauto."
      ["./cfrontend/Cminorgenproof.v", "Lemma padding_freeable_invariant"],
      ["./cfrontend/Cminorgenproof.v", "Lemma match_callstack_alloc_right"],
  ]
)
