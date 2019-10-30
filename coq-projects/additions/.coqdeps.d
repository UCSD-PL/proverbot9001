while.vo while.glob while.v.beautified: while.v
while.vio: while.v
two_power.vo two_power.glob two_power.v.beautified: two_power.v Constants.vo Mult_compl.vo Le_lt_compl.vo
two_power.vio: two_power.v Constants.vio Mult_compl.vio Le_lt_compl.vio
trivial.vo trivial.glob trivial.v.beautified: trivial.v monoid.vo
trivial.vio: trivial.v monoid.vio
strategies.vo strategies.glob strategies.v.beautified: strategies.v Constants.vo
strategies.vio: strategies.v Constants.vio
standard.vo standard.glob standard.v.beautified: standard.v monoid.vo
standard.vio: standard.v monoid.vio
spec.vo spec.glob spec.v.beautified: spec.v monoid.vo machine.vo Constants.vo
spec.vio: spec.v monoid.vio machine.vio Constants.vio
shift.vo shift.glob shift.v.beautified: shift.v Constants.vo Mult_compl.vo euclid.vo Le_lt_compl.vo
shift.vio: shift.v Constants.vio Mult_compl.vio euclid.vio Le_lt_compl.vio
monoid.vo monoid.glob monoid.v.beautified: monoid.v Constants.vo
monoid.vio: monoid.v Constants.vio
monofun.vo monofun.glob monofun.v.beautified: monofun.v monoid.vo
monofun.vio: monofun.v monoid.vio
matrix.vo matrix.glob matrix.v.beautified: matrix.v monoid.vo fmpc.vo
matrix.vio: matrix.v monoid.vio fmpc.vio
main.vo main.glob main.v.beautified: main.v Constants.vo generation.vo monoid.vo machine.vo strategies.vo spec.vo log2_spec.vo log2_implementation.vo while.vo imperative.vo develop.vo dicho_strat.vo binary_strat.vo trivial.vo standard.vo monofun.vo matrix.vo
main.vio: main.v Constants.vio generation.vio monoid.vio machine.vio strategies.vio spec.vio log2_spec.vio log2_implementation.vio while.vio imperative.vio develop.vio dicho_strat.vio binary_strat.vio trivial.vio standard.vio monofun.vio matrix.vio
machine.vo machine.glob machine.v.beautified: machine.v monoid.vo Constants.vo
machine.vio: machine.v monoid.vio Constants.vio
log2_spec.vo log2_spec.glob log2_spec.v.beautified: log2_spec.v Constants.vo Mult_compl.vo euclid.vo Le_lt_compl.vo shift.vo two_power.vo
log2_spec.vio: log2_spec.v Constants.vio Mult_compl.vio euclid.vio Le_lt_compl.vio shift.vio two_power.vio
log2_implementation.vo log2_implementation.glob log2_implementation.v.beautified: log2_implementation.v Constants.vo Mult_compl.vo euclid.vo Le_lt_compl.vo shift.vo imperative.vo while.vo two_power.vo log2_spec.vo
log2_implementation.vio: log2_implementation.v Constants.vio Mult_compl.vio euclid.vio Le_lt_compl.vio shift.vio imperative.vio while.vio two_power.vio log2_spec.vio
imperative.vo imperative.glob imperative.v.beautified: imperative.v
imperative.vio: imperative.v
generation.vo generation.glob generation.v.beautified: generation.v monoid.vo spec.vo Constants.vo machine.vo Le_lt_compl.vo euclid.vo shift.vo two_power.vo log2_spec.vo develop.vo strategies.vo Mult_compl.vo Wf_compl.vo
generation.vio: generation.v monoid.vio spec.vio Constants.vio machine.vio Le_lt_compl.vio euclid.vio shift.vio two_power.vio log2_spec.vio develop.vio strategies.vio Mult_compl.vio Wf_compl.vio
fmpc.vo fmpc.glob fmpc.v.beautified: fmpc.v
fmpc.vio: fmpc.v
euclid.vo euclid.glob euclid.v.beautified: euclid.v Constants.vo Le_lt_compl.vo Mult_compl.vo
euclid.vio: euclid.v Constants.vio Le_lt_compl.vio Mult_compl.vio
dicho_strat.vo dicho_strat.glob dicho_strat.v.beautified: dicho_strat.v strategies.vo euclid.vo shift.vo two_power.vo log2_spec.vo Le_lt_compl.vo Mult_compl.vo Constants.vo
dicho_strat.vio: dicho_strat.v strategies.vio euclid.vio shift.vio two_power.vio log2_spec.vio Le_lt_compl.vio Mult_compl.vio Constants.vio
develop.vo develop.glob develop.v.beautified: develop.v Constants.vo monoid.vo spec.vo machine.vo Wf_compl.vo Mult_compl.vo euclid.vo two_power.vo
develop.vio: develop.v Constants.vio monoid.vio spec.vio machine.vio Wf_compl.vio Mult_compl.vio euclid.vio two_power.vio
binary_strat.vo binary_strat.glob binary_strat.v.beautified: binary_strat.v strategies.vo euclid.vo shift.vo Le_lt_compl.vo Mult_compl.vo Constants.vo
binary_strat.vio: binary_strat.v strategies.vio euclid.vio shift.vio Le_lt_compl.vio Mult_compl.vio Constants.vio
Wf_compl.vo Wf_compl.glob Wf_compl.v.beautified: Wf_compl.v
Wf_compl.vio: Wf_compl.v
Mult_compl.vo Mult_compl.glob Mult_compl.v.beautified: Mult_compl.v Constants.vo Le_lt_compl.vo
Mult_compl.vio: Mult_compl.v Constants.vio Le_lt_compl.vio
Le_lt_compl.vo Le_lt_compl.glob Le_lt_compl.v.beautified: Le_lt_compl.v Constants.vo
Le_lt_compl.vio: Le_lt_compl.v Constants.vio
Constants.vo Constants.glob Constants.v.beautified: Constants.v
Constants.vio: Constants.v
extract.vo extract.glob extract.v.beautified: extract.v Constants.vo generation.vo monoid.vo machine.vo strategies.vo spec.vo log2_spec.vo log2_implementation.vo while.vo imperative.vo develop.vo dicho_strat.vo binary_strat.vo trivial.vo standard.vo monofun.vo matrix.vo main.vo
extract.vio: extract.v Constants.vio generation.vio monoid.vio machine.vio strategies.vio spec.vio log2_spec.vio log2_implementation.vio while.vio imperative.vio develop.vio dicho_strat.vio binary_strat.vio trivial.vio standard.vio monofun.vio matrix.vio main.vio
extract_hs.vo extract_hs.glob extract_hs.v.beautified: extract_hs.v Constants.vo generation.vo monoid.vo machine.vo strategies.vo spec.vo log2_spec.vo log2_implementation.vo while.vo imperative.vo develop.vo dicho_strat.vo binary_strat.vo trivial.vo standard.vo monofun.vo matrix.vo main.vo
extract_hs.vio: extract_hs.v Constants.vio generation.vio monoid.vio machine.vio strategies.vio spec.vio log2_spec.vio log2_implementation.vio while.vio imperative.vio develop.vio dicho_strat.vio binary_strat.vio trivial.vio standard.vio monofun.vio matrix.vio main.vio
extract_scm.vo extract_scm.glob extract_scm.v.beautified: extract_scm.v Constants.vo generation.vo monoid.vo machine.vo strategies.vo spec.vo log2_spec.vo log2_implementation.vo while.vo imperative.vo develop.vo dicho_strat.vo binary_strat.vo trivial.vo standard.vo monofun.vo matrix.vo main.vo
extract_scm.vio: extract_scm.v Constants.vio generation.vio monoid.vio machine.vio strategies.vio spec.vio log2_spec.vio log2_implementation.vio while.vio imperative.vio develop.vio dicho_strat.vio binary_strat.vio trivial.vio standard.vio monofun.vio matrix.vio main.vio
