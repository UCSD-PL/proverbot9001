bout_red.vo bout_red.glob bout_red.v.beautified: bout_red.v processus.vo induc.vo subtyping.vo typing_proofs.vo substitutions.vo fresh.vo out_red.vo
bout_red.vio: bout_red.v processus.vio induc.vio subtyping.vio typing_proofs.vio substitutions.vio fresh.vio out_red.vio
fresh.vo fresh.glob fresh.v.beautified: fresh.v processus.vo induc.vo
fresh.vio: fresh.v processus.vio induc.vio
induc.vo induc.glob induc.v.beautified: induc.v processus.vo
induc.vio: induc.v processus.vio
inp_red.vo inp_red.glob inp_red.v.beautified: inp_red.v processus.vo induc.vo subtyping.vo typing_proofs.vo substitutions.vo fresh.vo
inp_red.vio: inp_red.v processus.vio induc.vio subtyping.vio typing_proofs.vio substitutions.vio fresh.vio
out_red.vo out_red.glob out_red.v.beautified: out_red.v processus.vo induc.vo subtyping.vo typing_proofs.vo substitutions.vo fresh.vo
out_red.vio: out_red.v processus.vio induc.vio subtyping.vio typing_proofs.vio substitutions.vio fresh.vio
processus.vo processus.glob processus.v.beautified: processus.v
processus.vio: processus.v
substitutions.vo substitutions.glob substitutions.v.beautified: substitutions.v processus.vo fresh.vo
substitutions.vio: substitutions.v processus.vio fresh.vio
subtyping.vo subtyping.glob subtyping.v.beautified: subtyping.v processus.vo typing_proofs.vo swaps_proofs.vo fresh.vo substitutions.vo
subtyping.vio: subtyping.v processus.vio typing_proofs.vio swaps_proofs.vio fresh.vio substitutions.vio
swaps_proofs.vo swaps_proofs.glob swaps_proofs.v.beautified: swaps_proofs.v processus.vo fresh.vo typing_proofs.vo
swaps_proofs.vio: swaps_proofs.v processus.vio fresh.vio typing_proofs.vio
tau_red.vo tau_red.glob tau_red.v.beautified: tau_red.v processus.vo induc.vo subtyping.vo typing_proofs.vo substitutions.vo fresh.vo out_red.vo bout_red.vo inp_red.vo
tau_red.vio: tau_red.v processus.vio induc.vio subtyping.vio typing_proofs.vio substitutions.vio fresh.vio out_red.vio bout_red.vio inp_red.vio
typing_proofs.vo typing_proofs.glob typing_proofs.v.beautified: typing_proofs.v processus.vo fresh.vo
typing_proofs.vio: typing_proofs.v processus.vio fresh.vio
