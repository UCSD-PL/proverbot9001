gauss_int.vo gauss_int.glob gauss_int.v.beautified: gauss_int.v
gauss_int.vio: gauss_int.v
fermat2.vo fermat2.glob fermat2.v.beautified: fermat2.v gauss_int.vo
fermat2.vio: fermat2.v gauss_int.vio
