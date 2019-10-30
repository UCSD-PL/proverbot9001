ps.vo ps.glob ps.v.beautified: ps.v Ensembles.vo Relations_1.vo Relations_1_facts.vo podefs.vo podefs_1.vo
ps.vio: ps.v Ensembles.vio Relations_1.vio Relations_1_facts.vio podefs.vio podefs_1.vio
podefs_1.vo podefs_1.glob podefs_1.v.beautified: podefs_1.v Ensembles.vo Relations_1.vo podefs.vo
podefs_1.vio: podefs_1.v Ensembles.vio Relations_1.vio podefs.vio
podefs.vo podefs.glob podefs.v.beautified: podefs.v Ensembles.vo Relations_1.vo
podefs.vio: podefs.v Ensembles.vio Relations_1.vio
ex1_auto.vo ex1_auto.glob ex1_auto.v.beautified: ex1_auto.v
ex1_auto.vio: ex1_auto.v
ex1.vo ex1.glob ex1.v.beautified: ex1.v
ex1.vio: ex1.v
drinker.vo drinker.glob drinker.v.beautified: drinker.v
drinker.vio: drinker.v
Relations_3_facts.vo Relations_3_facts.glob Relations_3_facts.v.beautified: Relations_3_facts.v Relations_1.vo Relations_2.vo Relations_3.vo Relations_2_facts.vo
Relations_3_facts.vio: Relations_3_facts.v Relations_1.vio Relations_2.vio Relations_3.vio Relations_2_facts.vio
Relations_3.vo Relations_3.glob Relations_3.v.beautified: Relations_3.v Relations_1.vo Relations_2.vo
Relations_3.vio: Relations_3.v Relations_1.vio Relations_2.vio
Relations_2_facts.vo Relations_2_facts.glob Relations_2_facts.v.beautified: Relations_2_facts.v Relations_1.vo Relations_2.vo
Relations_2_facts.vio: Relations_2_facts.v Relations_1.vio Relations_2.vio
Relations_2.vo Relations_2.glob Relations_2.v.beautified: Relations_2.v Relations_1.vo
Relations_2.vio: Relations_2.v Relations_1.vio
Relations_1_facts.vo Relations_1_facts.glob Relations_1_facts.v.beautified: Relations_1_facts.v Relations_1.vo
Relations_1_facts.vio: Relations_1_facts.v Relations_1.vio
Relations_1.vo Relations_1.glob Relations_1.v.beautified: Relations_1.v
Relations_1.vio: Relations_1.v
Partial_order_facts.vo Partial_order_facts.glob Partial_order_facts.v.beautified: Partial_order_facts.v Ensembles.vo Relations_1.vo podefs.vo podefs_1.vo ps.vo
Partial_order_facts.vio: Partial_order_facts.v Ensembles.vio Relations_1.vio podefs.vio podefs_1.vio ps.vio
Fil.vo Fil.glob Fil.v.beautified: Fil.v Ensembles.vo Relations_1.vo podefs.vo podefs_1.vo ps.vo Partial_order_facts.vo
Fil.vio: Fil.v Ensembles.vio Relations_1.vio podefs.vio podefs_1.vio ps.vio Partial_order_facts.vio
Ensembles.vo Ensembles.glob Ensembles.v.beautified: Ensembles.v
Ensembles.vio: Ensembles.v
