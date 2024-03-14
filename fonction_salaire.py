def devine_salaire(genre, ville, age, profession,dep="") :
    
    import pandas as pd
    dfsal = pd.read_csv("net_salary_per_town_categories.csv")
    dfsal= dfsal.rename(columns={'SNHMFC14': 'Salaire_cadre_F',
                      "SNHMFP14":"Salaire_cadremoy_F",
                      "SNHMFE14":"Salaire_employe_F",
                      "SNHMFO14":"Salaire_travailleur_F",
                      "SNHMHC14":'Salaire_cadre_H',
                      "SNHMHP14":"Salaire_cadremoy_H",
                      "SNHMHE14":"Salaire_employe_H",
                      "SNHMHO14":"Salaire_travailleur_H",
                      "SNHMF1814":"Salaire_F_18-25",
                      "SNHMF2614":"Salaire_F_26-50",
                      "SNHMF5014":"Salaire_F_50",
                      "SNHMH1814":"Salaire_H_18-25",
                      "SNHMH2614":"Salaire_H_26-50",
                      "SNHMH5014":"Salaire_H_50",
                    "SNHM1814":"Salaire_ALL_18-25",
                             "SNHM2614":"Salaire_ALL_26-50",
                             "SNHM5014":"Sallaire_ALL_50",
                             "SNHMF14":"Salaire_F",
                             "SNHMH14":"Salaire_H",
                             "SNHM14":"Salaire_ALL",
                             "SNHMC14":"Salaire_ALL_cadre",
                             "SNHMP14":"Salaire_ALL_cadremoy",
                             "SNHME14":"Salaire_ALL_employe",
                             "SNHMO14":"Salaire_ALL_travailleur"
                     })
    doublon = dfsal[dfsal["LIBGEO"].duplicated()][["LIBGEO","CODGEO"]]
    doublon["dep"]=doublon.CODGEO.str.slice(0, 2)
    dfsal["dep"]=dfsal.CODGEO.str.slice(0, 2)
    
    ##cadre = 4
    ##cadre moyen = 3
    ##employe = 2
    ##travailleur = 1
    if ville in doublon.LIBGEO.values and dep=="":
        raise ValueError("Ville en doublon, précisez le numéro de département")
    elif ville in doublon.LIBGEO.values :
        CODGEO = dfsal[(dfsal.LIBGEO == ville)&(dfsal.dep == dep)].CODGEO.values[0]
        #print(CODGEO)
        ma_ville = dfsal[dfsal.CODGEO == CODGEO]
    else:
        ma_ville = dfsal[dfsal.LIBGEO == ville]
    
    #traitement des professions
    if profession == 4 :
        prof = "cadre"
    elif profession ==3 :
        prof = "cadremoy"
    elif profession ==2:
        prof = "employe"
    elif profession == 1 :
        prof = "travailleur"
    
    #traitement des ages
    if age <26 :
        groupe = "18-25"
    elif age <51:
        groupe = "26-50"
    elif age > 50 :
        groupe = "50"
        
   #création des noms de colonnes
    col1 = "Salaire_"+prof+"_"+genre
    col2 = "Salaire_"+genre+"_"+groupe
    
    ma_ville = ma_ville[[col1,col2]]

    #print(ma_ville)
    return((2*ma_ville.iloc[0,0]+ma_ville.iloc[0,1])/3)
