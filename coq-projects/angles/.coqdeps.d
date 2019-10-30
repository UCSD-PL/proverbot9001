point_angle.vo point_angle.glob point_angle.v.beautified: point_angle.v
point_angle.vio: point_angle.v
point_cocyclicite.vo point_cocyclicite.glob point_cocyclicite.v.beautified: point_cocyclicite.v point_angle.vo
point_cocyclicite.vio: point_cocyclicite.v point_angle.vio
point_tangente.vo point_tangente.glob point_tangente.v.beautified: point_tangente.v point_angle.vo
point_tangente.vio: point_tangente.v point_angle.vio
point_orthocentre.vo point_orthocentre.glob point_orthocentre.v.beautified: point_orthocentre.v point_cocyclicite.vo
point_orthocentre.vio: point_orthocentre.v point_cocyclicite.vio
point_napoleon.vo point_napoleon.glob point_napoleon.v.beautified: point_napoleon.v point_cocyclicite.vo
point_napoleon.vio: point_napoleon.v point_cocyclicite.vio
point_Simson.vo point_Simson.glob point_Simson.v.beautified: point_Simson.v point_cocyclicite.vo
point_Simson.vio: point_Simson.v point_cocyclicite.vio
