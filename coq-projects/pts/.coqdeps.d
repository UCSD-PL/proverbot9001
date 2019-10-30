Main.vo Main.glob Main.v.beautified: Main.v General.vo ./Ltermes.v ./Termes.v ./Env.v ./Subtyping_rule.v ./Ltyping.v ./Metatheory.v ./PTS_spec.v ./Infer.v ./Errors.v ./Lrules.v ./Rules.v ./Soundness.v ./PTS_spec.v ./Normal.v ./Confluence.v ./Conv.v ./Lcumul.v ./CTS_spec.v ./Cumul.v ./CumulDec.v ./CumulInfer.v ./Llambda.v ./Lambda_Rules.v ./LambdaSound.v ./PTS_spec.v ./BD.v ./Beta.v
Main.vio: Main.v General.vio ./Ltermes.v ./Termes.v ./Env.v ./Subtyping_rule.v ./Ltyping.v ./Metatheory.v ./PTS_spec.v ./Infer.v ./Errors.v ./Lrules.v ./Rules.v ./Soundness.v ./PTS_spec.v ./Normal.v ./Confluence.v ./Conv.v ./Lcumul.v ./CTS_spec.v ./Cumul.v ./CumulDec.v ./CumulInfer.v ./Llambda.v ./Lambda_Rules.v ./LambdaSound.v ./PTS_spec.v ./BD.v ./Beta.v
GenericSort.vo GenericSort.glob GenericSort.v.beautified: GenericSort.v
GenericSort.vio: GenericSort.v
SortECC.vo SortECC.glob SortECC.v.beautified: SortECC.v General.vo GenericSort.vo
SortECC.vio: SortECC.v General.vio GenericSort.vio
SortV6.vo SortV6.glob SortV6.v.beautified: SortV6.v General.vo GenericSort.vo
SortV6.vio: SortV6.v General.vio GenericSort.vio
CoqV6.vo CoqV6.glob CoqV6.v.beautified: CoqV6.v General.vo MyList.vo MyRelations.vo Main.vo SortV6.vo
CoqV6.vio: CoqV6.v General.vio MyList.vio MyRelations.vio Main.vio SortV6.vio
ECC.vo ECC.glob ECC.v.beautified: ECC.v General.vo MyList.vo MyRelations.vo Main.vo SortECC.vo
ECC.vio: ECC.v General.vio MyList.vio MyRelations.vio Main.vio SortECC.vio
CoqV6Beta.vo CoqV6Beta.glob CoqV6Beta.v.beautified: CoqV6Beta.v General.vo MyList.vo MyRelations.vo Main.vo SortV6.vo
CoqV6Beta.vio: CoqV6Beta.v General.vio MyList.vio MyRelations.vio Main.vio SortV6.vio
ExtractV6.vo ExtractV6.glob ExtractV6.v.beautified: ExtractV6.v MlExtract.vo General.vo CoqV6.vo
ExtractV6.vio: ExtractV6.v MlExtract.vio General.vio CoqV6.vio
MlExtract.vo MlExtract.glob MlExtract.v.beautified: MlExtract.v
MlExtract.vio: MlExtract.v
MyList.vo MyList.glob MyList.v.beautified: MyList.v
MyList.vio: MyList.v
MyRelations.vo MyRelations.glob MyRelations.v.beautified: MyRelations.v
MyRelations.vio: MyRelations.v
General.vo General.glob General.v.beautified: General.v MyList.vo MyRelations.vo
General.vio: General.v MyList.vio MyRelations.vio
