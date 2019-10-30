alloc.vo alloc.glob alloc.v.beautified: alloc.v misc.vo bool_fun.vo myMap.vo config.vo
alloc.vio: alloc.v misc.vio bool_fun.vio myMap.vio config.vio
bool_fun.vo bool_fun.glob bool_fun.v.beautified: bool_fun.v misc.vo
bool_fun.vio: bool_fun.v misc.vio
config.vo config.glob config.v.beautified: config.v misc.vo bool_fun.vo myMap.vo
config.vio: config.v misc.vio bool_fun.vio myMap.vio
gc.vo gc.glob gc.v.beautified: gc.v misc.vo bool_fun.vo config.vo myMap.vo
gc.vio: gc.v misc.vio bool_fun.vio config.vio myMap.vio
make.vo make.glob make.v.beautified: make.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo
make.vio: make.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio
misc.vo misc.glob misc.v.beautified: misc.v
misc.vio: misc.v
munew.vo munew.glob munew.v.beautified: munew.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo neg.vo or.vo univ.vo op.vo tauto.vo quant.vo gc.vo mu.vo
munew.vio: munew.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio neg.vio or.vio univ.vio op.vio tauto.vio quant.vio gc.vio mu.vio
muset.vo muset.glob muset.v.beautified: muset.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo neg.vo or.vo univ.vo op.vo tauto.vo quant.vo gc.vo mu.vo munew.vo
muset.vio: muset.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio neg.vio or.vio univ.vio op.vio tauto.vio quant.vio gc.vio mu.vio munew.vio
mu.vo mu.glob mu.v.beautified: mu.v myMap.vo misc.vo bool_fun.vo config.vo alloc.vo make.vo neg.vo or.vo univ.vo op.vo tauto.vo quant.vo gc.vo
mu.vio: mu.v myMap.vio misc.vio bool_fun.vio config.vio alloc.vio make.vio neg.vio or.vio univ.vio op.vio tauto.vio quant.vio gc.vio
myMap.vo myMap.glob myMap.v.beautified: myMap.v misc.vo
myMap.vio: myMap.v misc.vio
neg.vo neg.glob neg.v.beautified: neg.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo
neg.vio: neg.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio
op.vo op.glob op.v.beautified: op.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo neg.vo or.vo
op.vio: op.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio neg.vio or.vio
or.vo or.glob or.v.beautified: or.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo
or.vio: or.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio
quant.vo quant.glob quant.v.beautified: quant.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo or.vo op.vo tauto.vo gc.vo univ.vo
quant.vio: quant.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio or.vio op.vio tauto.vio gc.vio univ.vio
smc.vo smc.glob smc.v.beautified: smc.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo neg.vo or.vo univ.vo op.vo tauto.vo quant.vo gc.vo mu.vo munew.vo muset.vo
smc.vio: smc.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio neg.vio or.vio univ.vio op.vio tauto.vio quant.vio gc.vio mu.vio munew.vio muset.vio
tauto.vo tauto.glob tauto.v.beautified: tauto.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo neg.vo or.vo op.vo
tauto.vio: tauto.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio neg.vio or.vio op.vio
univ.vo univ.glob univ.v.beautified: univ.v misc.vo bool_fun.vo myMap.vo config.vo alloc.vo make.vo op.vo
univ.vio: univ.v misc.vio bool_fun.vio myMap.vio config.vio alloc.vio make.vio op.vio
