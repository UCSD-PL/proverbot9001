terminaison_SL.vo terminaison_SL.glob terminaison_SL.v.beautified: terminaison_SL.v sur_les_relations.vo TS.vo sigma_lift.vo comparith.vo Pol1.vo Pol2.vo
terminaison_SL.vio: terminaison_SL.v sur_les_relations.vio TS.vio sigma_lift.vio comparith.vio Pol1.vio Pol2.vio
sur_les_relations.vo sur_les_relations.glob sur_les_relations.v.beautified: sur_les_relations.v
sur_les_relations.vio: sur_les_relations.v
sigma_lift.vo sigma_lift.glob sigma_lift.v.beautified: sigma_lift.v TS.vo sur_les_relations.vo
sigma_lift.vio: sigma_lift.v TS.vio sur_les_relations.vio
resoudPC_SL.vo resoudPC_SL.glob resoudPC_SL.v.beautified: resoudPC_SL.v TS.vo sur_les_relations.vo sigma_lift.vo determinePC_SL.vo
resoudPC_SL.vio: resoudPC_SL.v TS.vio sur_les_relations.vio sigma_lift.vio determinePC_SL.vio
lambda_sigma_lift.vo lambda_sigma_lift.glob lambda_sigma_lift.v.beautified: lambda_sigma_lift.v TS.vo sur_les_relations.vo sigma_lift.vo
lambda_sigma_lift.vio: lambda_sigma_lift.v TS.vio sur_les_relations.vio sigma_lift.vio
inversionSL.vo inversionSL.glob inversionSL.v.beautified: inversionSL.v sur_les_relations.vo TS.vo sigma_lift.vo
inversionSL.vio: inversionSL.v sur_les_relations.vio TS.vio sigma_lift.vio
egaliteTS.vo egaliteTS.glob egaliteTS.v.beautified: egaliteTS.v TS.vo
egaliteTS.vio: egaliteTS.v TS.vio
determinePC_SL.vo determinePC_SL.glob determinePC_SL.v.beautified: determinePC_SL.v TS.vo sur_les_relations.vo egaliteTS.vo sigma_lift.vo inversionSL.vo
determinePC_SL.vio: determinePC_SL.v TS.vio sur_les_relations.vio egaliteTS.vio sigma_lift.vio inversionSL.vio
confluence_LSL.vo confluence_LSL.glob confluence_LSL.v.beautified: confluence_LSL.v TS.vo sur_les_relations.vo sigma_lift.vo lambda_sigma_lift.vo terminaison_SL.vo conf_local_SL.vo betapar.vo SLstar_bpar_SLstar.vo conf_strong_betapar.vo commutation.vo Newman.vo Yokouchi.vo
confluence_LSL.vio: confluence_LSL.v TS.vio sur_les_relations.vio sigma_lift.vio lambda_sigma_lift.vio terminaison_SL.vio conf_local_SL.vio betapar.vio SLstar_bpar_SLstar.vio conf_strong_betapar.vio commutation.vio Newman.vio Yokouchi.vio
conf_strong_betapar.vo conf_strong_betapar.glob conf_strong_betapar.v.beautified: conf_strong_betapar.v TS.vo sur_les_relations.vo betapar.vo egaliteTS.vo
conf_strong_betapar.vio: conf_strong_betapar.v TS.vio sur_les_relations.vio betapar.vio egaliteTS.vio
conf_local_SL.vo conf_local_SL.glob conf_local_SL.v.beautified: conf_local_SL.v TS.vo sur_les_relations.vo sigma_lift.vo determinePC_SL.vo resoudPC_SL.vo
conf_local_SL.vio: conf_local_SL.v TS.vio sur_les_relations.vio sigma_lift.vio determinePC_SL.vio resoudPC_SL.vio
comparith.vo comparith.glob comparith.v.beautified: comparith.v
comparith.vio: comparith.v
commutation.vo commutation.glob commutation.v.beautified: commutation.v sur_les_relations.vo TS.vo egaliteTS.vo sigma_lift.vo betapar.vo SLstar_bpar_SLstar.vo determinePC_SL.vo
commutation.vio: commutation.v sur_les_relations.vio TS.vio egaliteTS.vio sigma_lift.vio betapar.vio SLstar_bpar_SLstar.vio determinePC_SL.vio
betapar.vo betapar.glob betapar.v.beautified: betapar.v TS.vo sur_les_relations.vo
betapar.vio: betapar.v TS.vio sur_les_relations.vio
Yokouchi.vo Yokouchi.glob Yokouchi.v.beautified: Yokouchi.v sur_les_relations.vo
Yokouchi.vio: Yokouchi.v sur_les_relations.vio
TS.vo TS.glob TS.v.beautified: TS.v
TS.vio: TS.v
SLstar_bpar_SLstar.vo SLstar_bpar_SLstar.glob SLstar_bpar_SLstar.v.beautified: SLstar_bpar_SLstar.v TS.vo sur_les_relations.vo sigma_lift.vo lambda_sigma_lift.vo betapar.vo
SLstar_bpar_SLstar.vio: SLstar_bpar_SLstar.v TS.vio sur_les_relations.vio sigma_lift.vio lambda_sigma_lift.vio betapar.vio
Pol2.vo Pol2.glob Pol2.v.beautified: Pol2.v TS.vo sigma_lift.vo comparith.vo
Pol2.vio: Pol2.v TS.vio sigma_lift.vio comparith.vio
Pol1.vo Pol1.glob Pol1.v.beautified: Pol1.v TS.vo sigma_lift.vo comparith.vo
Pol1.vio: Pol1.v TS.vio sigma_lift.vio comparith.vio
Newman.vo Newman.glob Newman.v.beautified: Newman.v sur_les_relations.vo
Newman.vio: Newman.v sur_les_relations.vio
