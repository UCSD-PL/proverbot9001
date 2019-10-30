a_base.vo a_base.glob a_base.v.beautified: a_base.v
a_base.vio: a_base.v
b_soundness.vo b_soundness.glob b_soundness.v.beautified: b_soundness.v a_base.vo
b_soundness.vio: b_soundness.v a_base.vio
c_completeness.vo c_completeness.glob c_completeness.v.beautified: c_completeness.v b_soundness.vo
c_completeness.vio: c_completeness.v b_soundness.vio
d_hilbert_calculus.vo d_hilbert_calculus.glob d_hilbert_calculus.v.beautified: d_hilbert_calculus.v c_completeness.vo
d_hilbert_calculus.vio: d_hilbert_calculus.v c_completeness.vio
e_sequent_calculus.vo e_sequent_calculus.glob e_sequent_calculus.v.beautified: e_sequent_calculus.v c_completeness.vo
e_sequent_calculus.vio: e_sequent_calculus.v c_completeness.vio
f_cut_elimination.vo f_cut_elimination.glob f_cut_elimination.v.beautified: f_cut_elimination.v e_sequent_calculus.vo
f_cut_elimination.vio: f_cut_elimination.v e_sequent_calculus.vio
