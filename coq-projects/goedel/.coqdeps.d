checkPrf.vo checkPrf.glob checkPrf.v.beautified: checkPrf.v primRec.vo codeFreeVar.vo codeSubFormula.vo cPair.vo code.vo folProp.vo extEqualNat.vo wellFormed.vo folProof.vo prLogic.vo
checkPrf.vio: checkPrf.v primRec.vio codeFreeVar.vio codeSubFormula.vio cPair.vio code.vio folProp.vio extEqualNat.vio wellFormed.vio folProof.vio prLogic.vio
chRem.vo chRem.glob chRem.v.beautified: chRem.v
chRem.vio: chRem.v
codeFreeVar.vo codeFreeVar.glob codeFreeVar.v.beautified: codeFreeVar.v primRec.vo cPair.vo ListExt.vo codeList.vo folProp.vo code.vo
codeFreeVar.vio: codeFreeVar.v primRec.vio cPair.vio ListExt.vio codeList.vio folProp.vio code.vio
codeList.vo codeList.glob codeList.v.beautified: codeList.v primRec.vo cPair.vo ListExt.vo extEqualNat.vo
codeList.vio: codeList.v primRec.vio cPair.vio ListExt.vio extEqualNat.vio
codeNatToTerm.vo codeNatToTerm.glob codeNatToTerm.v.beautified: codeNatToTerm.v primRec.vo cPair.vo code.vo folProp.vo folProof.vo Languages.vo LNN.vo LNT.vo
codeNatToTerm.vio: codeNatToTerm.v primRec.vio cPair.vio code.vio folProp.vio folProof.vio Languages.vio LNN.vio LNT.vio
codePA.vo codePA.glob codePA.v.beautified: codePA.v primRec.vo cPair.vo folProp.vo code.vo codeList.vo codeFreeVar.vo extEqualNat.vo prLogic.vo PA.vo codeSubFormula.vo
codePA.vio: codePA.v primRec.vio cPair.vio folProp.vio code.vio codeList.vio codeFreeVar.vio extEqualNat.vio prLogic.vio PA.vio codeSubFormula.vio
codeSubFormula.vo codeSubFormula.glob codeSubFormula.v.beautified: codeSubFormula.v primRec.vo cPair.vo folProp.vo code.vo extEqualNat.vo codeSubTerm.vo codeFreeVar.vo
codeSubFormula.vio: codeSubFormula.v primRec.vio cPair.vio folProp.vio code.vio extEqualNat.vio codeSubTerm.vio codeFreeVar.vio
codeSubTerm.vo codeSubTerm.glob codeSubTerm.v.beautified: codeSubTerm.v primRec.vo cPair.vo folProp.vo code.vo extEqualNat.vo
codeSubTerm.vio: codeSubTerm.v primRec.vio cPair.vio folProp.vio code.vio extEqualNat.vio
codeSysPrf.vo codeSysPrf.glob codeSysPrf.v.beautified: codeSysPrf.v checkPrf.vo code.vo Languages.vo folProp.vo folProof.vo folLogic3.vo folReplace.vo PRrepresentable.vo expressible.vo primRec.vo PA.vo NNtheory.vo codeList.vo subProp.vo ListExt.vo cPair.vo wellFormed.vo prLogic.vo
codeSysPrf.vio: codeSysPrf.v checkPrf.vio code.vio Languages.vio folProp.vio folProof.vio folLogic3.vio folReplace.vio PRrepresentable.vio expressible.vio primRec.vio PA.vio NNtheory.vio codeList.vio subProp.vio ListExt.vio cPair.vio wellFormed.vio prLogic.vio
code.vo code.glob code.v.beautified: code.v fol.vo folProof.vo cPair.vo
code.vio: code.v fol.vio folProof.vio cPair.vio
cPair.vo cPair.glob cPair.v.beautified: cPair.v extEqualNat.vo primRec.vo
cPair.vio: cPair.v extEqualNat.vio primRec.vio
Deduction.vo Deduction.glob Deduction.v.beautified: Deduction.v folProof.vo folProp.vo
Deduction.vio: Deduction.v folProof.vio folProp.vio
expressible.vo expressible.glob expressible.v.beautified: expressible.v ListExt.vo folProp.vo subProp.vo extEqualNat.vo LNN.vo
expressible.vio: expressible.v ListExt.vio folProp.vio subProp.vio extEqualNat.vio LNN.vio
extEqualNat.vo extEqualNat.glob extEqualNat.v.beautified: extEqualNat.v
extEqualNat.vio: extEqualNat.v
fixPoint.vo fixPoint.glob fixPoint.v.beautified: fixPoint.v primRec.vo cPair.vo code.vo codeSubFormula.vo folProp.vo folProof.vo Languages.vo subAll.vo subProp.vo folLogic3.vo folReplace.vo LNN.vo codeNatToTerm.vo PRrepresentable.vo ListExt.vo NN.vo expressible.vo PA.vo NN2PA.vo
fixPoint.vio: fixPoint.v primRec.vio cPair.vio code.vio codeSubFormula.vio folProp.vio folProof.vio Languages.vio subAll.vio subProp.vio folLogic3.vio folReplace.vio LNN.vio codeNatToTerm.vio PRrepresentable.vio ListExt.vio NN.vio expressible.vio PA.vio NN2PA.vio
folLogic2.vo folLogic2.glob folLogic2.v.beautified: folLogic2.v ListExt.vo folProp.vo folProof.vo folLogic.vo subProp.vo folReplace.vo
folLogic2.vio: folLogic2.v ListExt.vio folProp.vio folProof.vio folLogic.vio subProp.vio folReplace.vio
folLogic3.vo folLogic3.glob folLogic3.v.beautified: folLogic3.v ListExt.vo folProp.vo folProof.vo folLogic2.vo subProp.vo folReplace.vo subAll.vo misc.vo
folLogic3.vio: folLogic3.v ListExt.vio folProp.vio folProof.vio folLogic2.vio subProp.vio folReplace.vio subAll.vio misc.vio
folLogic.vo folLogic.glob folLogic.v.beautified: folLogic.v ListExt.vo folProof.vo folProp.vo Deduction.vo
folLogic.vio: folLogic.v ListExt.vio folProof.vio folProp.vio Deduction.vio
folProof.vo folProof.glob folProof.v.beautified: folProof.v fol.vo folProp.vo
folProof.vio: folProof.v fol.vio folProp.vio
folProp.vo folProp.glob folProp.v.beautified: folProp.v ListExt.vo fol.vo
folProp.vio: folProp.v ListExt.vio fol.vio
folReplace.vo folReplace.glob folReplace.v.beautified: folReplace.v ListExt.vo folProof.vo folLogic.vo folProp.vo
folReplace.vio: folReplace.v ListExt.vio folProof.vio folLogic.vio folProp.vio
fol.vo fol.glob fol.v.beautified: fol.v misc.vo
fol.vio: fol.v misc.vio
goedel1.vo goedel1.glob goedel1.v.beautified: goedel1.v folProp.vo folProof.vo subProp.vo ListExt.vo fixPoint.vo codeSysPrf.vo wConsistent.vo NN.vo code.vo checkPrf.vo
goedel1.vio: goedel1.v folProp.vio folProof.vio subProp.vio ListExt.vio fixPoint.vio codeSysPrf.vio wConsistent.vio NN.vio code.vio checkPrf.vio
goedel2.vo goedel2.glob goedel2.v.beautified: goedel2.v folProp.vo folProof.vo folReplace.vo folLogic3.vo subProp.vo ListExt.vo fixPoint.vo NN2PA.vo codeSysPrf.vo PAtheory.vo code.vo checkPrf.vo codeNatToTerm.vo rosserPA.vo
goedel2.vio: goedel2.v folProp.vio folProof.vio folReplace.vio folLogic3.vio subProp.vio ListExt.vio fixPoint.vio NN2PA.vio codeSysPrf.vio PAtheory.vio code.vio checkPrf.vio codeNatToTerm.vio rosserPA.vio
Languages.vo Languages.glob Languages.v.beautified: Languages.v fol.vo primRec.vo
Languages.vio: Languages.v fol.vio primRec.vio
ListExt.vo ListExt.glob ListExt.v.beautified: ListExt.v
ListExt.vio: ListExt.v
LNN2LNT.vo LNN2LNT.glob LNN2LNT.v.beautified: LNN2LNT.v misc.vo ListExt.vo folProp.vo folProof.vo Languages.vo subAll.vo subProp.vo folLogic3.vo folReplace.vo LNT.vo codeNatToTerm.vo
LNN2LNT.vio: LNN2LNT.v misc.vio ListExt.vio folProp.vio folProof.vio Languages.vio subAll.vio subProp.vio folLogic3.vio folReplace.vio LNT.vio codeNatToTerm.vio
LNN.vo LNN.glob LNN.v.beautified: LNN.v Languages.vo folProof.vo folProp.vo folLogic3.vo
LNN.vio: LNN.v Languages.vio folProof.vio folProp.vio folLogic3.vio
LNT.vo LNT.glob LNT.v.beautified: LNT.v Languages.vo folProof.vo folProp.vo folLogic3.vo
LNT.vio: LNT.v Languages.vio folProof.vio folProp.vio folLogic3.vio
misc.vo misc.glob misc.v.beautified: misc.v
misc.vio: misc.v
model.vo model.glob model.v.beautified: model.v ListExt.vo folProof.vo folProp.vo misc.vo
model.vio: model.v ListExt.vio folProof.vio folProp.vio misc.vio
NN2PA.vo NN2PA.glob NN2PA.v.beautified: NN2PA.v folProp.vo folProof.vo subProp.vo folLogic3.vo folReplace.vo NN.vo PAtheory.vo LNN2LNT.vo subAll.vo ListExt.vo
NN2PA.vio: NN2PA.v folProp.vio folProof.vio subProp.vio folLogic3.vio folReplace.vio NN.vio PAtheory.vio LNN2LNT.vio subAll.vio ListExt.vio
NNtheory.vo NNtheory.glob NNtheory.v.beautified: NNtheory.v folLogic3.vo folProp.vo subProp.vo NN.vo
NNtheory.vio: NNtheory.v folLogic3.vio folProp.vio subProp.vio NN.vio
NN.vo NN.glob NN.v.beautified: NN.v folProp.vo subAll.vo folLogic3.vo Languages.vo LNN.vo
NN.vio: NN.v folProp.vio subAll.vio folLogic3.vio Languages.vio LNN.vio
PAconsistent.vo PAconsistent.glob PAconsistent.v.beautified: PAconsistent.v folProof.vo folProp.vo PA.vo model.vo
PAconsistent.vio: PAconsistent.v folProof.vio folProp.vio PA.vio model.vio
PAtheory.vo PAtheory.glob PAtheory.v.beautified: PAtheory.v subAll.vo folReplace.vo folProp.vo folLogic3.vo NN.vo LNN2LNT.vo PA.vo
PAtheory.vio: PAtheory.v subAll.vio folReplace.vio folProp.vio folLogic3.vio NN.vio LNN2LNT.vio PA.vio
PA.vo PA.glob PA.v.beautified: PA.v folProp.vo subAll.vo folLogic3.vo Languages.vo LNT.vo
PA.vio: PA.v folProp.vio subAll.vio folLogic3.vio Languages.vio LNT.vio
primRec.vo primRec.glob primRec.v.beautified: primRec.v extEqualNat.vo misc.vo
primRec.vio: primRec.v extEqualNat.vio misc.vio
prLogic.vo prLogic.glob prLogic.v.beautified: prLogic.v primRec.vo code.vo cPair.vo
prLogic.vio: prLogic.v primRec.vio code.vio cPair.vio
PRrepresentable.vo PRrepresentable.glob PRrepresentable.v.beautified: PRrepresentable.v extEqualNat.vo subAll.vo folProp.vo subProp.vo folReplace.vo folLogic3.vo NN.vo NNtheory.vo primRec.vo chRem.vo expressible.vo ListExt.vo cPair.vo
PRrepresentable.vio: PRrepresentable.v extEqualNat.vio subAll.vio folProp.vio subProp.vio folReplace.vio folLogic3.vio NN.vio NNtheory.vio primRec.vio chRem.vio expressible.vio ListExt.vio cPair.vio
rosserPA.vo rosserPA.glob rosserPA.v.beautified: rosserPA.v folProp.vo folProof.vo folReplace.vo folLogic3.vo subProp.vo ListExt.vo NNtheory.vo NN2PA.vo fixPoint.vo codeSysPrf.vo PAtheory.vo code.vo PRrepresentable.vo expressible.vo checkPrf.vo codeNatToTerm.vo codePA.vo PAconsistent.vo
rosserPA.vio: rosserPA.v folProp.vio folProof.vio folReplace.vio folLogic3.vio subProp.vio ListExt.vio NNtheory.vio NN2PA.vio fixPoint.vio codeSysPrf.vio PAtheory.vio code.vio PRrepresentable.vio expressible.vio checkPrf.vio codeNatToTerm.vio codePA.vio PAconsistent.vio
rosser.vo rosser.glob rosser.v.beautified: rosser.v folProp.vo folProof.vo folReplace.vo folLogic3.vo subProp.vo ListExt.vo fixPoint.vo codeSysPrf.vo NNtheory.vo code.vo PRrepresentable.vo expressible.vo checkPrf.vo codeNatToTerm.vo
rosser.vio: rosser.v folProp.vio folProof.vio folReplace.vio folLogic3.vio subProp.vio ListExt.vio fixPoint.vio codeSysPrf.vio NNtheory.vio code.vio PRrepresentable.vio expressible.vio checkPrf.vio codeNatToTerm.vio
subAll.vo subAll.glob subAll.v.beautified: subAll.v ListExt.vo folProof.vo folLogic2.vo folProp.vo folReplace.vo subProp.vo
subAll.vio: subAll.v ListExt.vio folProof.vio folLogic2.vio folProp.vio folReplace.vio subProp.vio
subProp.vo subProp.glob subProp.v.beautified: subProp.v ListExt.vo folProof.vo folLogic.vo folProp.vo folReplace.vo
subProp.vio: subProp.v ListExt.vio folProof.vio folLogic.vio folProp.vio folReplace.vio
wConsistent.vo wConsistent.glob wConsistent.v.beautified: wConsistent.v folProof.vo folProp.vo LNN.vo
wConsistent.vio: wConsistent.v folProof.vio folProp.vio LNN.vio
wellFormed.vo wellFormed.glob wellFormed.v.beautified: wellFormed.v primRec.vo cPair.vo code.vo folProp.vo extEqualNat.vo codeList.vo
wellFormed.vio: wellFormed.v primRec.vio cPair.vio code.vio folProp.vio extEqualNat.vio codeList.vio
