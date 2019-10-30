tactics.vo tactics.glob tactics.v.beautified: tactics.v
tactics.vio: tactics.v
missing.vo missing.glob missing.v.beautified: missing.v
missing.vio: missing.v
division.vo division.glob division.v.beautified: division.v missing.vo
division.vio: division.v missing.vio
gcd.vo gcd.glob gcd.v.beautified: gcd.v missing.vo division.vo euclide.vo power.vo
gcd.vio: gcd.v missing.vio division.vio euclide.vio power.vio
primes.vo primes.glob primes.v.beautified: primes.v missing.vo division.vo euclide.vo gcd.vo power.vo permutation.vo
primes.vio: primes.v missing.vio division.vio euclide.vio gcd.vio power.vio permutation.vio
power.vo power.glob power.v.beautified: power.v missing.vo division.vo
power.vio: power.v missing.vio division.vio
euclide.vo euclide.glob euclide.v.beautified: euclide.v missing.vo division.vo
euclide.vio: euclide.v missing.vio division.vio
nthroot.vo nthroot.glob nthroot.v.beautified: nthroot.v missing.vo division.vo gcd.vo primes.vo power.vo
nthroot.vio: nthroot.v missing.vio division.vio gcd.vio primes.vio power.vio
permutation.vo permutation.glob permutation.v.beautified: permutation.v missing.vo
permutation.vio: permutation.v missing.vio
