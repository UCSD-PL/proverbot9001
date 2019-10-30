divide.vo divide.glob divide.v.beautified: divide.v
divide.vio: divide.v
gcd.vo gcd.glob gcd.v.beautified: gcd.v divide.vo
gcd.vio: gcd.v divide.vio
prime.vo prime.glob prime.v.beautified: prime.v divide.vo gcd.vo
prime.vio: prime.v divide.vio gcd.vio
