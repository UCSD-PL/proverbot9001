ListSet.vo ListSet.glob ListSet.v.beautified: ListSet.v
ListSet.vio: ListSet.v
ListFunctions.vo ListFunctions.glob ListFunctions.v.beautified: ListFunctions.v ListSet.vo
ListFunctions.vio: ListFunctions.v ListSet.vio
SFSstate.vo SFSstate.glob SFSstate.v.beautified: SFSstate.v ListFunctions.vo
SFSstate.vio: SFSstate.v ListFunctions.vio
InitialState.vo InitialState.glob InitialState.v.beautified: InitialState.v SFSstate.vo
InitialState.vio: InitialState.v SFSstate.vio
sscstat.vo sscstat.glob sscstat.v.beautified: sscstat.v SFSstate.vo
sscstat.vio: sscstat.v SFSstate.vio
setACLdata.vo setACLdata.glob setACLdata.v.beautified: setACLdata.v SFSstate.vo
setACLdata.vio: setACLdata.v SFSstate.vio
DACandMAC.vo DACandMAC.glob DACandMAC.v.beautified: DACandMAC.v SFSstate.vo
DACandMAC.vio: DACandMAC.v SFSstate.vio
mkdir.vo mkdir.glob mkdir.v.beautified: mkdir.v DACandMAC.vo setACLdata.vo
mkdir.vio: mkdir.v DACandMAC.vio setACLdata.vio
create.vo create.glob create.v.beautified: create.v DACandMAC.vo setACLdata.vo
create.vio: create.v DACandMAC.vio setACLdata.vio
chmod.vo chmod.glob chmod.v.beautified: chmod.v DACandMAC.vo setACLdata.vo
chmod.vio: chmod.v DACandMAC.vio setACLdata.vio
write.vo write.glob write.v.beautified: write.v DACandMAC.vo
write.vio: write.v DACandMAC.vio
unlink.vo unlink.glob unlink.v.beautified: unlink.v DACandMAC.vo
unlink.vio: unlink.v DACandMAC.vio
stat.vo stat.glob stat.v.beautified: stat.v DACandMAC.vo
stat.vio: stat.v DACandMAC.vio
rmdir.vo rmdir.glob rmdir.v.beautified: rmdir.v DACandMAC.vo
rmdir.vio: rmdir.v DACandMAC.vio
readdir.vo readdir.glob readdir.v.beautified: readdir.v DACandMAC.vo
readdir.vio: readdir.v DACandMAC.vio
owner_close.vo owner_close.glob owner_close.v.beautified: owner_close.v DACandMAC.vo
owner_close.vio: owner_close.v DACandMAC.vio
oscstat.vo oscstat.glob oscstat.v.beautified: oscstat.v DACandMAC.vo
oscstat.vio: oscstat.v DACandMAC.vio
open.vo open.glob open.v.beautified: open.v DACandMAC.vo
open.vio: open.v DACandMAC.vio
close.vo close.glob close.v.beautified: close.v DACandMAC.vo
close.vio: close.v DACandMAC.vio
chsubsc.vo chsubsc.glob chsubsc.v.beautified: chsubsc.v DACandMAC.vo
chsubsc.vio: chsubsc.v DACandMAC.vio
chown.vo chown.glob chown.v.beautified: chown.v DACandMAC.vo
chown.vio: chown.v DACandMAC.vio
chobjsc.vo chobjsc.glob chobjsc.v.beautified: chobjsc.v DACandMAC.vo
chobjsc.vio: chobjsc.v DACandMAC.vio
addUsrGrpToAcl.vo addUsrGrpToAcl.glob addUsrGrpToAcl.v.beautified: addUsrGrpToAcl.v DACandMAC.vo
addUsrGrpToAcl.vio: addUsrGrpToAcl.v DACandMAC.vio
aclstat.vo aclstat.glob aclstat.v.beautified: aclstat.v DACandMAC.vo
aclstat.vio: aclstat.v DACandMAC.vio
read.vo read.glob read.v.beautified: read.v DACandMAC.vo open.vo
read.vio: read.v DACandMAC.vio open.vio
delUsrGrpFromAcl.vo delUsrGrpFromAcl.glob delUsrGrpFromAcl.v.beautified: delUsrGrpFromAcl.v DACandMAC.vo
delUsrGrpFromAcl.vio: delUsrGrpFromAcl.v DACandMAC.vio
TransFunc.vo TransFunc.glob TransFunc.v.beautified: TransFunc.v aclstat.vo chmod.vo create.vo mkdir.vo open.vo addUsrGrpToAcl.vo chobjsc.vo chown.vo chsubsc.vo close.vo delUsrGrpFromAcl.vo oscstat.vo owner_close.vo read.vo readdir.vo rmdir.vo sscstat.vo stat.vo unlink.vo write.vo
TransFunc.vio: TransFunc.v aclstat.vio chmod.vio create.vio mkdir.vio open.vio addUsrGrpToAcl.vio chobjsc.vio chown.vio chsubsc.vio close.vio delUsrGrpFromAcl.vio oscstat.vio owner_close.vio read.vio readdir.vio rmdir.vio sscstat.vio stat.vio unlink.vio write.vio
ModelProperties.vo ModelProperties.glob ModelProperties.v.beautified: ModelProperties.v TransFunc.vo
ModelProperties.vio: ModelProperties.v TransFunc.vio
AuxiliaryLemmas.vo AuxiliaryLemmas.glob AuxiliaryLemmas.v.beautified: AuxiliaryLemmas.v ModelProperties.vo
AuxiliaryLemmas.vio: AuxiliaryLemmas.v ModelProperties.vio
aclstatIsSecure.vo aclstatIsSecure.glob aclstatIsSecure.v.beautified: aclstatIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
aclstatIsSecure.vio: aclstatIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
addUsrGrpToAclIsSecure.vo addUsrGrpToAclIsSecure.glob addUsrGrpToAclIsSecure.v.beautified: addUsrGrpToAclIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
addUsrGrpToAclIsSecure.vio: addUsrGrpToAclIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
chmodIsSecure.vo chmodIsSecure.glob chmodIsSecure.v.beautified: chmodIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
chmodIsSecure.vio: chmodIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
chobjscIsSecure.vo chobjscIsSecure.glob chobjscIsSecure.v.beautified: chobjscIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
chobjscIsSecure.vio: chobjscIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
chownIsSecure.vo chownIsSecure.glob chownIsSecure.v.beautified: chownIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
chownIsSecure.vio: chownIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
chsubscIsSecure.vo chsubscIsSecure.glob chsubscIsSecure.v.beautified: chsubscIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
chsubscIsSecure.vio: chsubscIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
closeIsSecure.vo closeIsSecure.glob closeIsSecure.v.beautified: closeIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
closeIsSecure.vio: closeIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
createIsSecure.vo createIsSecure.glob createIsSecure.v.beautified: createIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
createIsSecure.vio: createIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
delUsrGrpFromAclIsSecure.vo delUsrGrpFromAclIsSecure.glob delUsrGrpFromAclIsSecure.v.beautified: delUsrGrpFromAclIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
delUsrGrpFromAclIsSecure.vio: delUsrGrpFromAclIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
mkdirIsSecure.vo mkdirIsSecure.glob mkdirIsSecure.v.beautified: mkdirIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
mkdirIsSecure.vio: mkdirIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
openIsSecure.vo openIsSecure.glob openIsSecure.v.beautified: openIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
openIsSecure.vio: openIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
oscstatIsSecure.vo oscstatIsSecure.glob oscstatIsSecure.v.beautified: oscstatIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
oscstatIsSecure.vio: oscstatIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
owner_closeIsSecure.vo owner_closeIsSecure.glob owner_closeIsSecure.v.beautified: owner_closeIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
owner_closeIsSecure.vio: owner_closeIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
readdirIsSecure.vo readdirIsSecure.glob readdirIsSecure.v.beautified: readdirIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
readdirIsSecure.vio: readdirIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
readIsSecure.vo readIsSecure.glob readIsSecure.v.beautified: readIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
readIsSecure.vio: readIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
rmdirIsSecure.vo rmdirIsSecure.glob rmdirIsSecure.v.beautified: rmdirIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
rmdirIsSecure.vio: rmdirIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
sscstatIsSecure.vo sscstatIsSecure.glob sscstatIsSecure.v.beautified: sscstatIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
sscstatIsSecure.vio: sscstatIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
statIsSecure.vo statIsSecure.glob statIsSecure.v.beautified: statIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
statIsSecure.vio: statIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
unlinkIsSecure.vo unlinkIsSecure.glob unlinkIsSecure.v.beautified: unlinkIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
unlinkIsSecure.vio: unlinkIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
writeIsSecure.vo writeIsSecure.glob writeIsSecure.v.beautified: writeIsSecure.v ModelProperties.vo AuxiliaryLemmas.vo
writeIsSecure.vio: writeIsSecure.v ModelProperties.vio AuxiliaryLemmas.vio
ModelLemmas.vo ModelLemmas.glob ModelLemmas.v.beautified: ModelLemmas.v ModelProperties.vo aclstatIsSecure.vo chmodIsSecure.vo createIsSecure.vo mkdirIsSecure.vo openIsSecure.vo addUsrGrpToAclIsSecure.vo chobjscIsSecure.vo chownIsSecure.vo chsubscIsSecure.vo closeIsSecure.vo delUsrGrpFromAclIsSecure.vo oscstatIsSecure.vo owner_closeIsSecure.vo readIsSecure.vo readdirIsSecure.vo rmdirIsSecure.vo sscstatIsSecure.vo statIsSecure.vo unlinkIsSecure.vo writeIsSecure.vo InitialState.vo
ModelLemmas.vio: ModelLemmas.v ModelProperties.vio aclstatIsSecure.vio chmodIsSecure.vio createIsSecure.vio mkdirIsSecure.vio openIsSecure.vio addUsrGrpToAclIsSecure.vio chobjscIsSecure.vio chownIsSecure.vio chsubscIsSecure.vio closeIsSecure.vio delUsrGrpFromAclIsSecure.vio oscstatIsSecure.vio owner_closeIsSecure.vio readIsSecure.vio readdirIsSecure.vio rmdirIsSecure.vio sscstatIsSecure.vio statIsSecure.vio unlinkIsSecure.vio writeIsSecure.vio InitialState.vio
