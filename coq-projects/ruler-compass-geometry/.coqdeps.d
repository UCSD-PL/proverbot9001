A1_Plan.vo A1_Plan.glob A1_Plan.v.beautified: A1_Plan.v
A1_Plan.vio: A1_Plan.v
A2_Orientation.vo A2_Orientation.glob A2_Orientation.v.beautified: A2_Orientation.v A1_Plan.vo
A2_Orientation.vio: A2_Orientation.v A1_Plan.vio
A3_Metrique.vo A3_Metrique.glob A3_Metrique.v.beautified: A3_Metrique.v A2_Orientation.vo
A3_Metrique.vio: A3_Metrique.v A2_Orientation.vio
A4_Droite.vo A4_Droite.glob A4_Droite.v.beautified: A4_Droite.v A3_Metrique.vo
A4_Droite.vio: A4_Droite.v A3_Metrique.vio
A5_Cercle.vo A5_Cercle.glob A5_Cercle.v.beautified: A5_Cercle.v A4_Droite.vo
A5_Cercle.vio: A5_Cercle.v A4_Droite.vio
A6_Intersection.vo A6_Intersection.glob A6_Intersection.v.beautified: A6_Intersection.v A5_Cercle.vo
A6_Intersection.vio: A6_Intersection.v A5_Cercle.vio
B10_Longueur_Prop.vo B10_Longueur_Prop.glob B10_Longueur_Prop.v.beautified: B10_Longueur_Prop.v B9_Inegalite_Triang.vo
B10_Longueur_Prop.vio: B10_Longueur_Prop.v B9_Inegalite_Triang.vio
B11_Angle_prop.vo B11_Angle_prop.glob B11_Angle_prop.v.beautified: B11_Angle_prop.v B10_Longueur_Prop.vo
B11_Angle_prop.vio: B11_Angle_prop.v B10_Longueur_Prop.vio
B12_Tacticques_base.vo B12_Tacticques_base.glob B12_Tacticques_base.v.beautified: B12_Tacticques_base.v B11_Angle_prop.vo
B12_Tacticques_base.vio: B12_Tacticques_base.v B11_Angle_prop.vio
B1_Confondu_Prop.vo B1_Confondu_Prop.glob B1_Confondu_Prop.v.beautified: B1_Confondu_Prop.v A6_Intersection.vo
B1_Confondu_Prop.vio: B1_Confondu_Prop.v A6_Intersection.vio
B2_Orient_Prop.vo B2_Orient_Prop.glob B2_Orient_Prop.v.beautified: B2_Orient_Prop.v B1_Confondu_Prop.vo
B2_Orient_Prop.vio: B2_Orient_Prop.v B1_Confondu_Prop.vio
B3_Alignes_Prop.vo B3_Alignes_Prop.glob B3_Alignes_Prop.v.beautified: B3_Alignes_Prop.v B2_Orient_Prop.vo
B3_Alignes_Prop.vio: B3_Alignes_Prop.v B2_Orient_Prop.vio
B4_Droite_Def.vo B4_Droite_Def.glob B4_Droite_Def.v.beautified: B4_Droite_Def.v B3_Alignes_Prop.vo
B4_Droite_Def.vio: B4_Droite_Def.v B3_Alignes_Prop.vio
B5_Entre_Prel.vo B5_Entre_Prel.glob B5_Entre_Prel.v.beautified: B5_Entre_Prel.v B4_Droite_Def.vo
B5_Entre_Prel.vio: B5_Entre_Prel.v B4_Droite_Def.vio
B6_Cercle_Def.vo B6_Cercle_Def.glob B6_Cercle_Def.v.beautified: B6_Cercle_Def.v B5_Entre_Prel.vo
B6_Cercle_Def.vio: B6_Cercle_Def.v B5_Entre_Prel.vio
B7_Triangle_Equilateral.vo B7_Triangle_Equilateral.glob B7_Triangle_Equilateral.v.beautified: B7_Triangle_Equilateral.v B6_Cercle_Def.vo
B7_Triangle_Equilateral.vio: B7_Triangle_Equilateral.v B6_Cercle_Def.vio
B8_Point_Def.vo B8_Point_Def.glob B8_Point_Def.v.beautified: B8_Point_Def.v B7_Triangle_Equilateral.vo
B8_Point_Def.vio: B8_Point_Def.v B7_Triangle_Equilateral.vio
B9_Inegalite_Triang.vo B9_Inegalite_Triang.glob B9_Inegalite_Triang.v.beautified: B9_Inegalite_Triang.v B8_Point_Def.vo
B9_Inegalite_Triang.vio: B9_Inegalite_Triang.v B8_Point_Def.vio
C10_Milieu.vo C10_Milieu.glob C10_Milieu.v.beautified: C10_Milieu.v C9_Triangles_Emboites.vo
C10_Milieu.vio: C10_Milieu.v C9_Triangles_Emboites.vio
C11_Mediatrice.vo C11_Mediatrice.glob C11_Mediatrice.v.beautified: C11_Mediatrice.v C10_Milieu.vo
C11_Mediatrice.vio: C11_Mediatrice.v C10_Milieu.vio
C12_Angles_Opposes.vo C12_Angles_Opposes.glob C12_Angles_Opposes.v.beautified: C12_Angles_Opposes.v C11_Mediatrice.vo
C12_Angles_Opposes.vio: C12_Angles_Opposes.v C11_Mediatrice.vio
C13_Angles_Supplem.vo C13_Angles_Supplem.glob C13_Angles_Supplem.v.beautified: C13_Angles_Supplem.v C12_Angles_Opposes.vo
C13_Angles_Supplem.vio: C13_Angles_Supplem.v C12_Angles_Opposes.vio
C14_Angle_Droit.vo C14_Angle_Droit.glob C14_Angle_Droit.v.beautified: C14_Angle_Droit.v C13_Angles_Supplem.vo
C14_Angle_Droit.vio: C14_Angle_Droit.v C13_Angles_Supplem.vio
C15_Parallelogramm.vo C15_Parallelogramm.glob C15_Parallelogramm.v.beautified: C15_Parallelogramm.v C14_Angle_Droit.vo
C15_Parallelogramm.vio: C15_Parallelogramm.v C14_Angle_Droit.vio
C1_DemiDroite_Prop.vo C1_DemiDroite_Prop.glob C1_DemiDroite_Prop.v.beautified: C1_DemiDroite_Prop.v B12_Tacticques_base.vo
C1_DemiDroite_Prop.vio: C1_DemiDroite_Prop.v B12_Tacticques_base.vio
C2_Entre_Prop.vo C2_Entre_Prop.glob C2_Entre_Prop.v.beautified: C2_Entre_Prop.v C1_DemiDroite_Prop.vo
C2_Entre_Prop.vio: C2_Entre_Prop.v C1_DemiDroite_Prop.vio
C3_Triangles_Egaux.vo C3_Triangles_Egaux.glob C3_Triangles_Egaux.v.beautified: C3_Triangles_Egaux.v C2_Entre_Prop.vo
C3_Triangles_Egaux.vio: C3_Triangles_Egaux.v C2_Entre_Prop.vio
C4_Triangles_non_degeneres_egaux.vo C4_Triangles_non_degeneres_egaux.glob C4_Triangles_non_degeneres_egaux.v.beautified: C4_Triangles_non_degeneres_egaux.v C3_Triangles_Egaux.vo
C4_Triangles_non_degeneres_egaux.vio: C4_Triangles_non_degeneres_egaux.v C3_Triangles_Egaux.vio
C5_Droite_Prop.vo C5_Droite_Prop.glob C5_Droite_Prop.v.beautified: C5_Droite_Prop.v C4_Triangles_non_degeneres_egaux.vo
C5_Droite_Prop.vio: C5_Droite_Prop.v C4_Triangles_non_degeneres_egaux.vio
C6_Parallele_Prop.vo C6_Parallele_Prop.glob C6_Parallele_Prop.v.beautified: C6_Parallele_Prop.v C5_Droite_Prop.vo
C6_Parallele_Prop.vio: C6_Parallele_Prop.v C5_Droite_Prop.vio
C7_DroitesSecantesProp.vo C7_DroitesSecantesProp.glob C7_DroitesSecantesProp.v.beautified: C7_DroitesSecantesProp.v C6_Parallele_Prop.vo
C7_DroitesSecantesProp.vio: C7_DroitesSecantesProp.v C6_Parallele_Prop.vio
C8_DroitesConfondues.vo C8_DroitesConfondues.glob C8_DroitesConfondues.v.beautified: C8_DroitesConfondues.v C7_DroitesSecantesProp.vo
C8_DroitesConfondues.vio: C8_DroitesConfondues.v C7_DroitesSecantesProp.vio
C9_Triangles_Emboites.vo C9_Triangles_Emboites.glob C9_Triangles_Emboites.v.beautified: C9_Triangles_Emboites.v C8_DroitesConfondues.vo
C9_Triangles_Emboites.vio: C9_Triangles_Emboites.v C8_DroitesConfondues.vio
D1_DistanceProp.vo D1_DistanceProp.glob D1_DistanceProp.v.beautified: D1_DistanceProp.v C15_Parallelogramm.vo
D1_DistanceProp.vio: D1_DistanceProp.v C15_Parallelogramm.vio
D2_Axe.vo D2_Axe.glob D2_Axe.v.beautified: D2_Axe.v D1_DistanceProp.vo
D2_Axe.vio: D2_Axe.v D1_DistanceProp.vio
D3_Triangle_Prop.vo D3_Triangle_Prop.glob D3_Triangle_Prop.v.beautified: D3_Triangle_Prop.v D2_Axe.vo
D3_Triangle_Prop.vio: D3_Triangle_Prop.v D2_Axe.vio
D4_CongruenceProp.vo D4_CongruenceProp.glob D4_CongruenceProp.v.beautified: D4_CongruenceProp.v D3_Triangle_Prop.vo
D4_CongruenceProp.vio: D4_CongruenceProp.v D3_Triangle_Prop.vio
D5_ParalleleConst.vo D5_ParalleleConst.glob D5_ParalleleConst.v.beautified: D5_ParalleleConst.v D4_CongruenceProp.vo
D5_ParalleleConst.vio: D5_ParalleleConst.v D4_CongruenceProp.vio
D6_SumAnglesProp.vo D6_SumAnglesProp.glob D6_SumAnglesProp.v.beautified: D6_SumAnglesProp.v D5_ParalleleConst.vo
D6_SumAnglesProp.vio: D6_SumAnglesProp.v D5_ParalleleConst.vio
D7_NonParalleles_Secantes.vo D7_NonParalleles_Secantes.glob D7_NonParalleles_Secantes.v.beautified: D7_NonParalleles_Secantes.v D6_SumAnglesProp.vo
D7_NonParalleles_Secantes.vio: D7_NonParalleles_Secantes.v D6_SumAnglesProp.vio
E1_Incidence.vo E1_Incidence.glob E1_Incidence.v.beautified: E1_Incidence.v D7_NonParalleles_Secantes.vo
E1_Incidence.vio: E1_Incidence.v D7_NonParalleles_Secantes.vio
E2_Ordre.vo E2_Ordre.glob E2_Ordre.v.beautified: E2_Ordre.v E1_Incidence.vo
E2_Ordre.vio: E2_Ordre.v E1_Incidence.vio
E3_Congruence.vo E3_Congruence.glob E3_Congruence.v.beautified: E3_Congruence.v E2_Ordre.vo
E3_Congruence.vio: E3_Congruence.v E2_Ordre.vio
E4_Continuite.vo E4_Continuite.glob E4_Continuite.v.beautified: E4_Continuite.v E3_Congruence.vo
E4_Continuite.vio: E4_Continuite.v E3_Congruence.vio
E5_Parallelisme.vo E5_Parallelisme.glob E5_Parallelisme.v.beautified: E5_Parallelisme.v E4_Continuite.vo
E5_Parallelisme.vio: E5_Parallelisme.v E4_Continuite.vio
