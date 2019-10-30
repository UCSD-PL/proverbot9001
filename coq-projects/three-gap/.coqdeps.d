Nat_compl.vo Nat_compl.glob Nat_compl.v.beautified: Nat_compl.v
Nat_compl.vio: Nat_compl.v
tools.vo tools.glob tools.v.beautified: tools.v Nat_compl.vo
tools.vio: tools.v Nat_compl.vio
prop_elem.vo prop_elem.glob prop_elem.v.beautified: prop_elem.v tools.vo
prop_elem.vio: prop_elem.v tools.vio
prop_fl.vo prop_fl.glob prop_fl.v.beautified: prop_fl.v prop_elem.vo
prop_fl.vio: prop_fl.v prop_elem.vio
preuve1.vo preuve1.glob preuve1.v.beautified: preuve1.v prop_fl.vo
preuve1.vio: preuve1.v prop_fl.vio
preuve2.vo preuve2.glob preuve2.v.beautified: preuve2.v preuve1.vo
preuve2.vio: preuve2.v preuve1.vio
three_gap.vo three_gap.glob three_gap.v.beautified: three_gap.v preuve2.vo
three_gap.vio: three_gap.v preuve2.vio
