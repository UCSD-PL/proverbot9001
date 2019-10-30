Examples.vo Examples.glob Examples.v.beautified: Examples.v Maple.vo
Examples.vio: Examples.v Maple.vio
Maple.vo Maple.glob Maple.v.beautified: Maple.v ./maple$(DYNOBJ)
Maple.vio: Maple.v ./maple$(DYNOBJ)
