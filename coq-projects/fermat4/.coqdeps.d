ArithCompl.vo ArithCompl.glob ArithCompl.v.beautified: ArithCompl.v
ArithCompl.vio: ArithCompl.v
Tactics.vo Tactics.glob Tactics.v.beautified: Tactics.v ArithCompl.vo
Tactics.vio: Tactics.v ArithCompl.vio
Pythagorean.vo Pythagorean.glob Pythagorean.v.beautified: Pythagorean.v Tactics.vo
Pythagorean.vio: Pythagorean.v Tactics.vio
Descent.vo Descent.glob Descent.v.beautified: Descent.v
Descent.vio: Descent.v
Diophantus20.vo Diophantus20.glob Diophantus20.v.beautified: Diophantus20.v Descent.vo Pythagorean.vo
Diophantus20.vio: Diophantus20.v Descent.vio Pythagorean.vio
Fermat4.vo Fermat4.glob Fermat4.v.beautified: Fermat4.v Diophantus20.vo
Fermat4.vio: Fermat4.v Diophantus20.vio
