misc.vo misc.glob misc.v.beautified: misc.v Lci.vo
misc.vio: misc.v Lci.vio
Lci.vo Lci.glob Lci.v.beautified: Lci.v
Lci.vio: Lci.v
rings.vo rings.glob rings.v.beautified: rings.v Lci.vo misc.vo groups.vo
rings.vio: rings.v Lci.vio misc.vio groups.vio
groups.vo groups.glob groups.v.beautified: groups.v Lci.vo misc.vo
groups.vio: groups.v Lci.vio misc.vio
Zstruct.vo Zstruct.glob Zstruct.v.beautified: Zstruct.v Lci.vo misc.vo groups.vo rings.vo
Zstruct.vio: Zstruct.v Lci.vio misc.vio groups.vio rings.vio
Zgcd.vo Zgcd.glob Zgcd.v.beautified: Zgcd.v misc.vo Zstruct.vo
Zgcd.vio: Zgcd.v misc.vio Zstruct.vio
