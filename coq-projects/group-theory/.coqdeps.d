Group_definitions.vo Group_definitions.glob Group_definitions.v.beautified: Group_definitions.v Laws.vo
Group_definitions.vio: Group_definitions.v Laws.vio
g2.vo g2.glob g2.v.beautified: g2.v Laws.vo Group_definitions.vo gr.vo g1.vo
g2.vio: g2.v Laws.vio Group_definitions.vio gr.vio g1.vio
gr.vo gr.glob gr.v.beautified: gr.v Laws.vo Group_definitions.vo
gr.vio: gr.v Laws.vio Group_definitions.vio
g3.vo g3.glob g3.v.beautified: g3.v ./Z/Zbase.vo ./Z/Z_succ_pred.vo ./Z/Zadd.vo ./Z/Zle.vo Laws.vo Group_definitions.vo gr.vo g1.vo
g3.vio: g3.v ./Z/Zbase.vio ./Z/Z_succ_pred.vio ./Z/Zadd.vio ./Z/Zle.vio Laws.vio Group_definitions.vio gr.vio g1.vio
./Z/Zle.vo ./Z/Zle.glob ./Z/Zle.v.beautified: ./Z/Zle.v ./Z/Zbase.vo ./Z/Z_succ_pred.vo ./Z/Zadd.vo
./Z/Zle.vio: ./Z/Zle.v ./Z/Zbase.vio ./Z/Z_succ_pred.vio ./Z/Zadd.vio
./Z/Zbase.vo ./Z/Zbase.glob ./Z/Zbase.v.beautified: ./Z/Zbase.v
./Z/Zbase.vio: ./Z/Zbase.v
./Z/Nat_complements.vo ./Z/Nat_complements.glob ./Z/Nat_complements.v.beautified: ./Z/Nat_complements.v
./Z/Nat_complements.vio: ./Z/Nat_complements.v
./Z/Z_succ_pred.vo ./Z/Z_succ_pred.glob ./Z/Z_succ_pred.v.beautified: ./Z/Z_succ_pred.v ./Z/Zbase.vo
./Z/Z_succ_pred.vio: ./Z/Z_succ_pred.v ./Z/Zbase.vio
./Z/Zadd.vo ./Z/Zadd.glob ./Z/Zadd.v.beautified: ./Z/Zadd.v ./Z/Nat_complements.vo ./Z/Zbase.vo ./Z/Z_succ_pred.vo
./Z/Zadd.vio: ./Z/Zadd.v ./Z/Nat_complements.vio ./Z/Zbase.vio ./Z/Z_succ_pred.vio
Laws.vo Laws.glob Laws.v.beautified: Laws.v
Laws.vio: Laws.v
Relations.vo Relations.glob Relations.v.beautified: Relations.v
Relations.vio: Relations.v
g1.vo g1.glob g1.v.beautified: g1.v Laws.vo Group_definitions.vo gr.vo ./Z/Zbase.vo ./Z/Z_succ_pred.vo ./Z/Zadd.vo
g1.vio: g1.v Laws.vio Group_definitions.vio gr.vio ./Z/Zbase.vio ./Z/Z_succ_pred.vio ./Z/Zadd.vio
