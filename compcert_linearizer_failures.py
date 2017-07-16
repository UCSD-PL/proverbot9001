compcert_failures = (
  [ ["./backend/Stackingproof.v", "Lemma set_location"], # fails because of linearizing "eapply Mem.load_store_similar_2; eauto."
    ["./cfrontend/Cminorgenproof.v", "Lemma padding_freeable_invariant"],
    ["./cfrontend/Cminorgenproof.v", "Lemma match_callstack_alloc_right"],
  ]
)
