# EDA — Constats (bloc 1 « Evidence »)

Format : constat -> chiffre -> conséquence pour la modélisation.
Généré depuis `notebooks/01_eda.ipynb` ; tous les chiffres sont calculés.

## Qualité des données
- Déséquilibre de classe -> prévalence 5.99% (IC Wilson [5.62%, 6.37%]) -> métrique PR-AUC + rappel@budget, jamais l'accuracy (baseline 'toujours No' = 94.01%).
- PolicyNumber unique, croissant, aligné aux années -> ID + proxy temporel -> retirer des features ET l'utiliser comme axe du split.
- Age==0 sur 320 lignes, toutes 'AgeOfPolicyHolder=16 to 17' (fraude 9.69%) -> sentinelle de saisie -> traiter en catégorie/NaN, pas âge 0.
- 1 ligne avec date de déclaration '0' -> saisie invalide -> passée en NaN au chargement.
- 0 doublon une fois PolicyNumber retiré -> chaque ligne est un sinistre distinct.
- BasePolicy entièrement déterminée par PolicyType ; 4 987 'Sedan - Liability' toutes 'Sport' -> redondance + incohérence -> ne garder qu'une des deux, écarter/reconstruire VehicleCategory.

## Association (taille d'effet)
- 30 tests chi², 19 significatifs après BH ; à n=15,420 commenter l'effet, pas l'étoile.
- PolicyType : V de Cramér = 0.167 (p_BH = 5.5e-88) -> signal exploitable, effet modéré.
- BasePolicy : V de Cramér = 0.161 (p_BH = 5e-87) -> signal exploitable, effet modéré.
- VehicleCategory : V de Cramér = 0.137 (p_BH = 6.6e-63) -> signal exploitable, effet modéré.
- Fault : V de Cramér = 0.131 (p_BH = 5.7e-59) -> signal exploitable, effet modéré.
- AddressChange-Claim : V de Cramér = 0.081 (p_BH = 5.8e-21) -> signal exploitable, effet modéré.
- Deductible : V de Cramér = 0.067 (p_BH = 6.5e-15) -> signal exploitable, effet modéré.
- AddressChange='under 6 months' (4 lignes) et Deductible=300 (8 lignes) -> n<30, IC large -> non concluant -> regrouper (fit sur train).

## Dérive temporelle
- Taux 1994=6.66%, 1995=5.79%, 1996=5.22% ; Cochran-Armitage z=-3.07, p=0.0021 -> baisse significative de 1.44 pts -> split TEMPOREL obligatoire.
- Saisonnalité par mois : V de Cramér=0.035 (p_BH=0.0034) -> effet faible, IC chevauchants -> pas de traitement spécifique.

## Ordinales à encoder (ordre métier)
- 17 variables de plage stockées en texte -> encodage ordinal (pas one-hot, pas alphabétique) : AddressChange-Claim, AgeOfPolicyHolder, AgeOfVehicle, DayOfWeek, DayOfWeekClaimed, Days:Policy-Accident, Days:Policy-Claim, Deductible, DriverRating, Month, MonthClaimed, NumberOfCars, NumberOfSuppliments, PastNumberOfClaims, VehiclePrice, WeekOfMonth, WeekOfMonthClaimed.
