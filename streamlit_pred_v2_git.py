import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle
import plotly.express as px
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.api
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import requests
from io import BytesIO
import geopandas as gpd

dfent = pd.read_csv("base_etablissement_par_tranche_effectif.csv", sep = ",") 
dfgeo = pd.read_csv ("name_geographic_information.csv", sep = ",")
dfsal = pd.read_csv("net_salary_per_town_categories.csv", sep = ",")

st.title("💶 French Industry 💶")
st.sidebar.title("Sommaire")
pages=["👋 Introduction","🔍 Exploration des données", "📊 DataViz",'⚙️ Pre-processing & 🧠 Modèles de Machine Learning',"🔮 Prédictions","📌Conclusion"]
page=st.sidebar.radio("⬇️ Aller vers", pages)
st.sidebar.markdown(
    """
    - **Cursus** : Data Analyst
    - **Formation** : Bootcamp
    - **Mois** : Jan.2024
    - **Groupe** : 
        - Yannick Lin
        - Manon Péquillat
        - Margaux Polomack""")

# Page d'introduction
if page == pages[0] :
    # Présentation projet
    st.caption("""**Cursus** : Data Analyst
    | **Formation** : Bootcamp
    | **Mois** : Janvier 2024""")

    st.header("👋 Introduction")
    st.markdown("""<style>h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    st.write("Notre projet porte sur l'analyse de la disparité des salaires en France et sur les critères qui peuvent influencer son montant. Grâce à nos prédictions, nous vous donnerons alors les critères qui vous permettra de l'augmenter.")
    st.write("")
    with st.expander("Commençons"):
    # Question et options de réponse
        question = "À votre avis, comment gagner un salaire haut en France ?"
        options = [
            "En habitant en Ile de France ?",
            "En étant cadre supérieur ?",
            "En étant un homme ?",
            "En ayant des enfants ?",
            "Toutes les réponses !"
        ]

        # Affichage de la question et des options
        st.title("Comment bien gagner sa vie ?")
        st.write(question)
        reponse = st.radio("Choisissez une réponse :", options, index=None)
        
        # Affichage du bouton pour soumettre la réponse
        if reponse:
            # Affichage du message après avoir choisi une réponse
            st.write("<div style='font-size: 30px; text-align: center';>🥁 C'est ce que nous allons voir dans cette présentation ! 🥁</div>", unsafe_allow_html=True)

# Page d'exploration des données
if page == pages[1] : 
    df_full = pd.read_csv("df_full.csv")
    df_full_ratio_commune = pd.read_csv("df_full_ratio_commune.csv")
    df_full_ratio_dept = pd.read_csv("df_full_ratio_dept.csv")
    st.header("🔍 Exploration des Données")
    st.markdown(""" """ """<style>h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown(""" """ """<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    if st.button("Entreprises") :
        #Afficher le Dataset dfent
        st.subheader("Dataset sur la répartition des entreprises en fonction des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfent.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfent.shape[0])
        st.write("- Nombre de colonnes:", dfent.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfent.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfent.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfent.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)
      
            st.write("")

    if st.button("Nom géographique") :
        #Afficher le Dataset dfgeo
        st.subheader("Dataset sur les informations géographiques des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfgeo.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfgeo.shape[0])
        st.write("- Nombre de colonnes:", dfgeo.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfgeo.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfgeo.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfgeo.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)

        st.write("")

    if st.button("Salaires") :
        #Afficher le Dataset dfsal
        st.subheader("Dataset sur la répartition des salaires moyens en fonction des communes françaises et différentes caractéristiques")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfsal.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", dfsal.shape[0])
        st.write("- Nombre de colonnes:", dfsal.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfsal.describe())
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfsal.isnull().sum())
        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            dfsal.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)

        st.write("")

    if st.button("Population") :
        #Afficher le Dataset dfent
        dfpop_light = pd.read_csv("dfpop_light.csv")
        dfpop_missing_values = pd.read_csv("dfpop_missing_values.csv")
        dfpop_describe = pd.read_csv("dfpop_describe.csv")

        st.subheader("Dataset sur les caractéristiques de la population en fonction des communes françaises")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(dfpop_light)
        nb_lignes_pop = 8536584
        nb_col_pop = 7
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", nb_lignes_pop)
        st.write("- Nombre de colonnes:", nb_col_pop)
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(dfpop_describe)
        st.write("")
        col1, col2 = st.columns([2,3])
        with col1 : 
            st.write("- Valeurs manquantes :", dfpop_missing_values)
        with col2:
            st.write("- Informations :")
            st.image("popinfo.png")

        st.write("")



# Page de Datavisualisation
if page == pages[2] :
    
    df_full = pd.read_csv("df_full.csv")
    df_full_ratio_commune = pd.read_csv("df_full_ratio_commune.csv")
    df_full_ratio_dept = pd.read_csv("df_full_ratio_dept.csv")
    st.markdown("""<style>h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    st.header("📊 Data Visualisation")
    # Distribution du salaire moyen
    st.subheader("1. Distribution du salaire moyen")
    fig = px.histogram(dfsal, x='SNHM14', nbins=10, marginal='box', color_discrete_sequence=px.colors.sequential.Viridis, labels={'SNHM14': 'Salaire horaire net moyen'})
    fig.update_layout(title="Distribution du salaire moyen avec boxplot",
                    xaxis_title="Salaire horaire net moyen",
                    yaxis_title="Fréquence",
                    showlegend=False)
    st.plotly_chart(fig)

    st.write("")

    st.markdown("Nous pouvons voir que le montant horaire du salaire moyen en France se situe entre 10 et 15€. Les salaires moyen supérieurs à 18/20€ sont faibles et ressortent comme outliers dans le boxplot.")

    st.write("")
    st.write("")
    st.write("")

    st.subheader("2. Répartition des entreprises et des salaires en fonction des régions françaises")
  
    # Regroupement du nombre d'entreprises en fonction de la région
    dfent_REG = df_full.groupby("REG").agg({"E14TST":"sum", "nom_région":"max"}).reset_index(drop=False)
    dfent_REG.columns = ["REG", "nombre_entreprises", "nom_région"]
    geo_region = gpd.read_file('regions.geojson')
    # Carte de la répartition des entreprises par régions
    fig = px.choropleth(dfent_REG, 
                    geojson=geo_region, 
                    locations="nom_région", 
                    featureidkey="properties.nom",  
                    color="nombre_entreprises",
                    hover_name="nom_région",
                    title="Nombre d'entreprises par région en France",
                    color_continuous_scale=px.colors.sequential.Viridis)
    
    # Ajuste la vue pour la France
    fig.update_geos(fitbounds="locations", visible=False,projection_scale=5)  
    #fig.update_layout(height=800, width=1000)
    st.plotly_chart(fig)

    st.write("")

    # Graphique de la répartition du nombre d'entreprises par régions
    fig = px.bar(dfent_REG, x='nom_région', y='nombre_entreprises', title="Répartition du nombre d'entreprises par régions", color='nom_région', color_discrete_sequence=px.colors.sequential.Viridis, labels={'nombre_entreprises': "Nombre d'entreprises"})
    fig.update_layout(xaxis_title="Régions", yaxis_title="Nombre d'entreprises", showlegend=False)
    fig.update_xaxes(tickangle=45)

    # Afficher les graphiques
    st.plotly_chart(fig)

    # Graphique des salaires moyens par régions
    fig = px.box(df_full.sort_values(by="REG"), x='nom_région', y='SNHM14', title='Salaires moyens par régions', color='nom_région', color_discrete_sequence=px.colors.sequential.Viridis, labels={'SNHM14': 'Salaire Moyen'})
    fig.update_layout(xaxis_title="Régions", yaxis_title="Salaire Moyen", showlegend=False)
    fig.update_xaxes(tickangle=45)

    # Afficher les graphiques
    st.plotly_chart(fig)

    st.write("")

    paragraph = ("Nous constatons que la région Ile-de-France comporte le plus grand nombre d’entreprises. "
             "Nous remarquons que la répartition des entreprises en France est très inégale avec une concentration de celles-ci en Ile-de-France. "
             "Ensuite, ce sont les régions Auvergne-Rhône-Alpes et Provence-Côte d’Azur qui suivent. "
             "La région comptant le moins d’entreprises est la Corse. <br>"
             "La répartition des salaires est également inégale. En effet, la région Ile-de-France compte les communes disposant des plus hauts salaires avec un salaire médian de 16€/h alors que les autres régions disposent des médianes autour de 12.5€/h. "
             "Nous pouvons aussi constater que la région Auvergne-Rhône-Alpes possède des communes avec des salaires moyens très élevés.")

    st.markdown(paragraph, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")

    st.subheader("3. Répartition des entreprises et des salaires en fonction des départements français")

    # Regroupement du nombre d'entreprises en fonction du département
    df_DEP = df_full.groupby("DEP").agg({"nom_département":"max", "REG":"max","E14TST":"sum", "SNHM14" : ["mean","min","max"]}).reset_index(drop=False)
    df_DEP.columns = ["DEP", "nom_département", "REG", "Nombre_entreprises", "salaire moyen", "salaire min","salaire max"]
    
    # Carte de la répartition des entreprises par départements
    fig = px.choropleth(df_DEP, 
                        geojson=gpd.read_file("departements.geojson"), 
                        locations="nom_département", 
                        featureidkey="properties.nom",  
                        color="Nombre_entreprises",
                        hover_name="nom_département",
                        title="Nombre d'entreprises par départements français",
                        color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_geos(fitbounds="locations", visible=False,projection_scale=5)  # Ajuste la vue pour la France
    #fig.update_layout(height=800, width=1000)
    st.plotly_chart(fig)

    st.write("")

    # Carte de la répartition du salaire moyen par départements
    fig = px.choropleth(df_DEP, 
                        geojson=gpd.read_file("departements.geojson"), 
                        locations="nom_département", 
                        featureidkey="properties.nom",  
                        color="salaire moyen",
                        hover_name="nom_département",
                        title="Salaire moyen par départements français",
                        color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_geos(fitbounds="locations", visible=False,projection_scale=5)  # Ajuste la vue pour la France
    #fig.update_layout(height=800, width=1000)
    st.plotly_chart(fig)

    # Graphique de la répartition du nombre d'entreprises et des salaires en IDF
    df_idf = df_DEP[df_DEP["REG"] == 11]
    # Création du graphique pour l'Île-de-France (Barplot)
    fig = px.bar(df_idf, x='nom_département', y='Nombre_entreprises', title='Nombre d\'entreprises en IDF', text='Nombre_entreprises', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Nombre d'entreprises", showlegend=False)
    # Afficher les graphiques
    st.plotly_chart(fig)

    # Création du graphique pour l'Île-de-France (Boxplot)
    fig = px.box(df_idf, x='nom_département', y='salaire moyen', title='Salaires moyens en IDF', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Salaire Moyen", showlegend=False)
    # Affichage des graphiques
    st.plotly_chart(fig)

    st.write("")

    # Graphique de la répartition du nombre d'entreprises et des salaires en PACA
    df_paca = df_DEP[df_DEP["REG"] == 93]
    # Création du graphique pour la PACA (Barplot)
    fig = px.bar(df_paca, x='nom_département', y='Nombre_entreprises', title='Nombre d\'entreprises en PACA', text='Nombre_entreprises', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Nombre d'entreprises", showlegend=False)
    # Afficher les graphiques
    st.plotly_chart(fig)

    # Création du graphique pour la PACA (Boxplot)
    df_paca_box = df_full[df_full["REG"]==93]
    fig = px.box(df_paca_box, x='nom_département', y='SNHM14', title='Salaires moyens en PACA', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Salaire Moyen", showlegend=False)
    # Affichage des graphiques
    st.plotly_chart(fig)

    st.write("")

    # Graphique de la répartition du nombre d'entreprises et des salaires en ARA
    df_ara = df_DEP[df_DEP["REG"] == 84]
    # Graphique pour l'Auvergne-Rhône-Alpes (Barplot)
    fig = px.bar(df_ara, x='nom_département', y='Nombre_entreprises', text='Nombre_entreprises', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Nombre d'entreprises", showlegend=False)
    # Afficher les graphiques
    st.plotly_chart(fig)

    # Graphique pour l'Auvergne-Rhône-Alpes (Boxplot)
    fig = px.box(df_ara, x='nom_département', y='salaire moyen', color='DEP', color_discrete_sequence=px.colors.sequential.Viridis, labels={'DEP' : 'département'})
    fig.update_layout(xaxis_title="Départements", yaxis_title="Salaire Moyen", showlegend=False)
    # Afficher les graphiques
    st.plotly_chart(fig)

    st.write("")   

    st.markdown(""""
        Nous constatons que la localisation a un impact sur le salaire moyen.\n 
        - En effet, en Ile-de-France (IDF), nous remarquons que la quasi-totalité  des entreprises se situent dans la capitale (Paris - 75).\n 
        - En Provence-Alpes-Côte d’Azur (PACA), la répartition des entreprises est très hétérogène avec une concentration des entreprises dans les départements avec les villes de Marseille (13) et de Nice (06).\n 
        - En Auvergne-Rhône-Alpes (ARA), nous remarquons que le département de Lyon (69) concentre une grande partie des entreprises. Mais la région dispose aussi d’autres départements qui attirent les entreprises avec le Puy-de-Dôme (63), l’Isère (38) et la Loire (42) qui rassemblent plus de 10 000 entreprises.\n 
                """)
    
    st.write("")
    st.write("")
    st.write("")

    st.subheader("4. Comparaison des salaires entre les hommes et les femmes en France")
    # Créer la figure avec deux subplots côte à côte
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Salaire moyen hommes', 'Salaire moyen femmes'])
    # Boxplot pour le salaire moyen des hommes
    fig1 = px.box(df_full, y='SNHMH14', color_discrete_sequence=['#440154'])
    fig1.update_layout(yaxis_title='Salaire moyen hommes', yaxis_range=[9, 50])
    # Ajouter le premier graphique à la première colonne du subplot
    fig.add_trace(fig1.data[0], row=1, col=1)
    # Boxplot pour le salaire moyen des femmes
    fig2 = px.box(df_full, y='SNHMF14', color_discrete_sequence=['#21918C'])
    fig2.update_layout(yaxis_title='Salaire moyen femmes', yaxis_range=[9, 50])
    # Ajouter le deuxième graphique à la deuxième colonne du subplot
    fig.add_trace(fig2.data[0], row=1, col=2)
    # Ajuster la figure
    fig.update_layout(height=500, width=800, showlegend=False)
    fig.update_yaxes(range=[9, 50], row=1, col=2)
    fig.update_yaxes(range=[9, 50], row=1, col=1)
    # Afficher la figure
    st.plotly_chart(fig)

    st.write("")   

    st.markdown("Nous remarquons une inégalité des salaires moyens selon le sexe de l’individu. Les femmes ont des salaires globalement plus faibles que les hommes.")
    
    st.write("")
    st.write("")
    st.write("")

    st.subheader("5. Corrélations sur le salaire moyen")

    # Sélectionner les colonnes pour la matrice de corrélation
    columns = ["salaire_moyen", "IDF", "densite_commune", "ent_total", 
            "ratio_TPE", "ratio_PME", "ratio_ETI_GE", "ratio_autre_ent", 
            "pop_totale", "age_moyen",
            "ratio_homme", "ratio_femme",
            "ratio_enfant", "ratio_jeune_actif", "ratio_actif", "ratio_senior_actif", "ratio_senior", 
            "ratio_solo", "ratio_e2p", "ratio_ase", "ratio_e1p", "ratio_ac", "ratio_ace", "ratio_str"]
    # Créer la matrice de corrélation
    correlation_matrix = df_full_ratio_commune[columns].corr()
    # Créer la heatmap avec Plotly Express
    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Corrélation"),
                    x=columns,
                    y=columns,
                    color_continuous_scale="Viridis")
    # Mise en page et affichage de la figure
    fig.update_layout(title="Matrice de corrélation des ratios à l'échelle de la commune",
                    width=800, height=800)
    st.plotly_chart(fig)

    st.write("")

    st.markdown("""
                Les corrélations sont assez faibles entre le salaire moyen et les variables explicatives.  \n 
                Les variables les plus corrélées positivement sont : 
                - la présence de la commune en Ile-de-France (IDF) 
                - la représentation des foyers vivant en couple avec enfant (ratio_ace) 
                - la représentation de personnes entre 35 et 55 ans (ratio_actif)
                - la représentation d'enfants vivant avec leurs deux parents (ratio_e2p)
                - la densité de la commune (densite_commune).
                 \n 
                Les variables plutôt corrélées négativement sont : la présence d’une grande proportion de très petites entreprises (ratio_ETI_GE) ; et de personnes habitant seules (ratio_solo).
                 \n 
                Comme le département semble avoir la plus grande importance dans nos corrélations, nous décidons de regrouper notre jeu de données par département et d’observer une nouvelle matrice de corrélation. Cette fois, nous ajoutons de nouvelles données :
                - la densité exacte du département (densite)
                - le taux de chômage (chomage)
                - la part de jeunes diplômés sur le marché du travail (part_jeunedip_pas_etudes)
                - la part de jeunes non diplômés (part_1624_pas_etudes).
                    """)

    st.write("")
    st.write("")
    st.write("")

    # Créer la matrice de corrélation
    correlation_matrix_full_dept = df_full_ratio_dept[["salaire_moyen", "ent_total", "pop_totale", "densite","ratio_TPE", "ratio_PME", "ratio_ETI_GE","ratio_homme", "ratio_femme","ratio_enfant", "ratio_jeune_actif", "ratio_actif", "ratio_senior_actif", "ratio_senior","ratio_solo", "ratio_e2p", "ratio_ase", "ratio_e1p", "ratio_ac", "ratio_ace", "ratio_str","part_1624_pasetudes", "part_jeunedip_pasetudes", "chomage"]].corr()
    # Créer une figure heatmap avec Plotly Express
    fig = px.imshow(correlation_matrix_full_dept,
                    labels=dict(color="Corrélation"),
                    color_continuous_scale='Viridis')
    # Ajouter des annotations et affichage de la figure
    fig.update_layout(
        title="Matrice de corrélation des ratios à l'échelle du département",
        xaxis=dict(title="Variables"),
        yaxis=dict(title="Variables"),
        coloraxis_colorbar=dict(title="Corrélation"),
        width=800, height=800
    )
    st.plotly_chart(fig)

    st.write("")

    st.markdown("""
                Les corrélations sont beaucoup plus marquées.  \n 
                Les variables les plus corrélées positivement sont : la part de jeunes diplômés sur le marché du travail, le nombre d'entreprises, la population totale, la représentation des 25-35 ans, la densité.  \n  
                Les variables plutôt corrélées négativement sont : la part de 55-60 ans, les plus de 60 ans et les très petites entreprises.
                    """)

    st.write("")
    st.write("")
    st.write("")


# Présentation de la modélisation

if page == pages[3] :


    df1 = pd.read_csv("df_ratio.csv", index_col="CODGEO")
    df2 = pd.read_csv("df_full_cadre_tertiaire.csv", index_col="CODGEO")
    df3 = pd.read_csv("df_full_cadre_tertiaire_tag_dep.csv", index_col="CODGEO")
    df4 = pd.read_csv("df_cat_poste.csv", index_col="CODGEO")
    st.header("⚙️ Pre-processing & 🧠 Modèles de Machine Learning")
    st.markdown("""<style>h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    
    st.subheader("Objectif")
    st.write("Prédire le salaire moyen d'une personne (variable continue) en fonction des données contenues dans les variables explicatives qui le composent.")
    
    st.write("")
    st.write("")

    if st.button("Modèles de régression") :
        st.subheader("Choix des modèles")
        st.markdown("""
                    Afin de prédire le salaire moyen, nous avons étudié la performance de plusieurs modèles de machine learning :
                    - Régression linéaire
                    - Ridge
                    - Lasso
                    - Ridge
                    - Elastic Net
                    - KNN
                    - Decision Tree
                    - Random Forest
                    - SVR.

                    Ces modèles ont été appliqués à 4 Dataframes issus de pré-processing différent afin de comparer les résultats et de déterminer le meilleur modèle.
        """)

        st.write("")
        st.write("")

    # Présentation du jeu de modélisation n°1
        
    if st.button("Choix du jeu de modélisation"):
        st.subheader("Jeu de modélisation n°1")
        st.markdown("""
                    Ce DataFrame a été créé à partir des données fournies pour le projet et avec les variables (ensemble de ratios sur la population et les entreprises) créées lors de l’initialisation des matrices de corrélation
                    """)
        st.write("")
        st.write("")
        st.dataframe(df1.head())

        st.markdown("#### Tableau global des résultats")
        result1 = pd.read_csv("result_model1.csv", index_col = 1)

        result1.drop(columns=result1.columns[0], axis=1, inplace=True)

        st.dataframe(result1)

        st.markdown("""
                    #### Interprétation


                    On constate des R² plutôt faibles (autour de 0.4, voire 0.2) ou du sur apprentissage lorsque le R² est meilleur (pour Random Forest et SVR).  
                    
                    
                    La MSE sur l’ensemble de test est assez élevée. La RMSE de test, à son minimum est à 1.42 (donc un écart moyen de 1.42€ sur le salaire moyen horaire prédit).  
                    
                    
                    Pour ce premier jeu de modélisation, on peine donc à trouver un “meilleur” modèle. Nous souhaitons donc améliorer notre jeu de données.  
                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°2
        
        st.subheader("Jeu de modélisation n°2")
        st.markdown("""
                    Sur la base du DataFrame précédent, nous avons ajouté :
                    - La proportion de cadre par commune en 2020
                    - La proportion de personnes travaillant dans le tertiaire par commune en 2020

                    """)
        st.write("")
        st.write("")
        st.dataframe(df2.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats

        result2 = pd.read_csv("result_model2.csv", index_col = 1)

        result2.drop(columns=result2.columns[0], axis=1, inplace=True)
        st.dataframe(result2)

        st.markdown("""
                    #### Interprétation
                    Les modèles KNN et SVR semblent les plus intéressants (les écarts entre R² train et R² test sont les plus faibles, tout en étant proche de 1). Les écarts de prédictions (RMSE respectivement à 1.28 et 1.11).
                    
                    
                    Le Random Forest a une RMSE faible mais l’écart entre les valeurs de R² témoignent d’un surapprentissage.
                    
                    
                    Afin d'affiner ces résultats, nous effectuons un GridSearch pour évaluer les meilleurs paramètres à passer pour chacun de ces modèles.

                    """)
        st.write("")
        st.write("")

        st.markdown("#### Résultats après Gridsearch")
        # Insertion résultat gridsearch
        data = {
        'Model': ['KNN', 'SRV'],
        'R2_train': [0.859765, 0.885198],
        'R2_test': [0.681018, 0.802175],
        'R2_full': [0.815537,0.864396],
        'MAE_train': [0.628312, 0.772503],
        'MAE_test': [0.934093, 1.180191],
        'MSE_train': [0.912875, 0.476711],
        'MSE_test': [2.114212, 0.663887],
        'RMSE_train': [0.955445, 0.878921],
        'RMSE_test': [1.454033, 1.086366]
                }
        
        tab = pd.DataFrame(data)
        tab.set_index('Model', inplace = True)
        st.dataframe(tab)

        st.markdown("""
                    Le modèle KNN a un surapprentissage beaucoup trop important. Le modèle SVR a une meilleure erreur quadratique, sans pour autant perdre la qualité de son écart entre r2 train et test.

                    
                    On constate en observant l’importance des variables explicatives dans les différents modèles que la proportion de cadres est désormais la variable la plus importante.

                    
                    Est-il possible d’améliorer nos scores avec l’ajout d’une donnée sur le département d’appartenance (comme pour l’appartenance à la région Ile-de-France) ?


                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°3

        st.subheader("Jeu de modélisation n°3")
        st.markdown("""
                    Sur base du DataFrame précédent, nous avons ajouté une colonne par département encodé selon si la commune fait partie (1) ou non (0) à ce département (OneHotEncoder).
                            
                            """)

        st.write("")
        st.write("")
        st.dataframe(df3.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats
        result3 = pd.read_csv("result_model3.csv", index_col = 1)

        result3.drop(columns=result3.columns[0], axis=1, inplace=True)

        st.dataframe(result3)

        st.markdown("""
                    #### Interprétation
                    Les résultats ne sont pas sensiblement meilleurs. Le SVR est un plus grand surapprentissage que le jeu de données précédent, sans pour autant améliorer l’erreur quadratique.
                    """)
        st.write("")
        st.write("")


    # Présentation du jeu de modélisation n°4
        
        st.subheader("Jeu de modélisation n°4")
        st.markdown("""
                    Ce 4e DataFrame a été créé depuis le fichier net_salary.csv fourni pour le projet. Les colonnes ont été pivoté afin de catégoriser le salaire moyen selon le genre et la catégorie de poste :
                    - genre = 1 : femme / 2 : homme
                    - cat_poste = 1 : travailleur(euse) / 2 : employé(e) / 3 : cadre moyen / 4 : cadre

                    """)
        st.write("")
        st.write("")
        st.dataframe(df4.head())

        st.markdown("#### Tableau global des résultats")
        # Insertion du dataframe résultats
        result4 = pd.read_csv("result_model4.csv", index_col = 1)

        result4.drop(columns=result4.columns[0], axis=1, inplace=True)

        st.dataframe(result4)

        st.markdown("""
                    #### Interprétation
                    On note un R² de test supérieur au R² train sur le modèle SVR. Cependant, les valeurs des RMSE de test sont moins intéressantes que sur le DataFrame n°2.

                    
                    Il est intéressant de noter que pour SVR, nous avons un DataFrame de grandeur optimale d’après la courbe d’apprentissage. 

                    
                    Le modèle SVR offre les meilleurs résultats. Si on analyse de plus près le graphique de dispersions des résidus de ce modèle, on constate de nombreux résidus avec une valeur parfois supérieure à 5, ce qui est très important pour un salaire horaire.

                    """)
        st.write("")
        st.write("")
    
        st.subheader("Modèle le plus performant")

        st.markdown("""
                    En conclusion, le jeu de données avec la proportion de cadre semble offrir le meilleur compromis en termes de performance (jeu de modélisation n°2 avec le modèle SVR).

                     
                    Les modèles Random Forest obtiennent de bons résultats d’erreur quadratique, malheureusement, l’écart des R² est trop important signe de surapprentissage.

                    """)
        st.write("")
        st.write("")

    if st.button("Evaluation graphique du modèle") :
        
        #Courbe d'apprentissage
        st.subheader("Courbe d'apprentissage du modèle")
        # insérer la courbe d'apprentissage

        image_apprentissage = "https://zupimages.net/up/24/11/m3js.png"

        st.image(image_apprentissage, use_column_width=True)

        st.markdown("""
                    ##### Points à retenir :         
                    - Modèle qui s'ajuste au fur à mesure aux données d'entrainement.
                    - Pas de soupçon d'overfitting
                    """)
        
        st.write("")
        st.write("")

        #QQplot, residu et prediction vs vraies
        st.subheader("Graphique des résidus et QQ-plot")
        # insérer qqplot et résidu
        image_qqplot = "https://zupimages.net/up/24/11/n6me.png"

        st.image(image_qqplot, use_column_width=True)

        st.write("")
        st.write("")

        st.markdown("""
                    ##### Points à retenir :      
                    - Distribution relativement centrée autour de 0 et entre -2.5 et 2.5
                    - Quelques valeurs extrêmes.

                    """)
        
        st.write("")
        st.write("")


    if st.button("Features importance") :
        
        st.markdown("""
                    Le SRV n'ayant pas de Features importance, nous allons présenter ceux du RandomForestRegressor, un modèle qui donne aussi des résultats très satisfaisant.
                    """
                    )

        # Visualiser les importances des caractéristiques
        st.subheader("Importance des variables du RandomForestRegressor")
        
        # insérer le graphe
        image_importances = "https://zupimages.net/up/24/11/82nd.png"

        st.image(image_importances, use_column_width=True)
        st.markdown("""
                    On remarque que la variable la plus importante est le ratio_cadre qui représente 70% pour le modèle.
                    """)
        
if page == pages[4] :
    df=pd.read_csv("df_full_full_catpos.csv")
    st.image('pred_banner.png', use_column_width=True)
    st.write('### Qui a le salaire le plus élevé ?')
    #st.image('regis_banner.png', use_column_width=True)
    #st.image('daphne_banner.png', use_column_width=True)
    commune_chargee = "Ambérieu-en-Bugey"
    genre_charge = "H"
    cat_chargee = "Ouvrier"
    age_charge = 18

    with st.expander("Cas n° 1"):
    # Ajoutez du contenu à l'intérieur de la zone dépliante
        #st.write("<p style='font-size:20px;'>Régis</p>", unsafe_allow_html=True)
        st.write("<div style='text-align: center;font-size:50px;'>👨🏽‍💼5️⃣5️⃣ 👨🏽‍💻 👑</div>", unsafe_allow_html=True)
        checkbox_1 = st.checkbox("Qui est cette personne ?",key="checkbox_1")
        # Si la case à cocher est cochée, afficher du texte supplémentaire
        if checkbox_1:
            st.write("Régis est un CTO de 55 ans qui vit à Versailles dans les Yvelines")
            if st.button("Charger les caractéristiques de Régis"):
            # Charger la valeur "Versailles" dans le champ de sélection
                commune_chargee = "Versailles"
                genre_charge = "H"
                age_charge = 55
                cat_chargee = "Cadre supérieur"

    
    with st.expander("Cas n° 2"):
    # Ajoutez du contenu à l'intérieur de la zone dépliante
        #st.write("<p style='font-size:20px;'>Régis</p>", unsafe_allow_html=True)
        st.write("<div style='text-align: center;font-size:50px;'>🙋‍♀️2️⃣7️⃣ 👷‍♀️ 🌞</div>", unsafe_allow_html=True)
        checkbox_2 = st.checkbox("Qui est cette personne ?",key="checkbox_2")
        # Si la case à cocher est cochée, afficher du texte supplémentaire
        if checkbox_2:
            st.write("Daphné a 27 et est ouvrière sur des chantiers, elle vit à Montastruc-la-Conseillère en Haute-Garronne")
            if st.button("Charger les caractéristiques de Daphné"):
            # Charger la valeur "Versailles" dans le champ de sélection
                commune_chargee = "Montastruc-la-Conseillère"
                genre_charge = "F"
                age_charge = 27
                cat_chargee = "Ouvrier"

    st.write('### Entrez les caractéristiques de la personne')

    df = pd.read_csv('df_full_full_propcadre.csv')
    df.drop_duplicates(inplace=True)
    df.rename(columns = {'LIBGEO' : 'nom_commune'}, inplace = True)
    dfcad = pd.read_csv('prop_cadre.csv')
    dfcad.drop_duplicates(subset=["nom_commune"], keep= 'first', inplace= True)
    dfcad1 = df.merge(dfcad, on = 'nom_commune', how = 'left')
    dfter = pd.read_csv('prop_tertiaire.csv')
    dfter.rename(columns = {'LIBGEO' : 'nom_commune'}, inplace = True)
    dfter.drop_duplicates(subset=["nom_commune"], keep= 'first', inplace= True)
    dftot = dfcad1.merge(dfter, on = 'nom_commune', how = 'left')
    dftot.dropna(inplace = True)
    df = dftot
    df.DEP = df.DEP.replace("2A","222")
    df.DEP = df.DEP.replace("2B","223")
    df_debase = df
    df = df.drop(['nom_commune',"CODGEO",'nom_DEP','nom_REG','nom_commune',"Departement"],axis=1)
    df_debase = df_debase.drop(['nom_DEP','nom_REG',"Departement",'salaire_moyen'],axis=1)
    X = df.drop('salaire_moyen', axis=1) 
    y = df['salaire_moyen']

    def charger_modele():
    # Charger le modèle à partir du fichier Pickle
        with open('modele2.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele

    liste_ville = df_debase["nom_commune"].drop_duplicates().tolist()
    liste_genre = ["H","F"]
    liste_cat = ["Cadre supérieur", "Cadre","Travailleur","Ouvrier"]
    commune = st.selectbox('Ville', options=liste_ville, index=liste_ville.index(commune_chargee))
    genre = st.selectbox('Sexe',options=liste_genre, index=liste_genre.index(genre_charge))
    age = st.number_input('Votre âge', min_value= 18 ,step = 1,value=age_charge)
    cat_poste = st.selectbox('Type Poste',options=liste_cat,index=liste_cat.index(cat_chargee) )
    if cat_poste=="Cadre supérieur":
        cat_poste = "cadre sup"
    if cat_poste=="Cadre":
        cat_poste="cadre"
    if cat_poste=="Travailleur" :
        cat_poste = "travailleur"
    if cat_poste =="Ouvrier":
        cat_poste="ouvrier"
    ma_ligne = df_debase.loc[df_debase["nom_commune"]==commune]

    #traitement du genre
    if genre != "":
        if genre == "F" :
            ma_ligne["ratio_homme"] = 0
            ma_ligne["ratio_femme"] = 1
        elif genre =="H":
            ma_ligne["ratio_femme"] = 0
            ma_ligne["ratio_homme"] = 1
        
    ma_ligne = df_debase.loc[df_debase["nom_commune"]==commune]
 
    ma_ligne["age_moyen"]=age

    ma_ligne=ma_ligne.drop(["nom_commune","CODGEO"],axis=1)    
    X = pd.concat([X, ma_ligne], ignore_index=True)

    scaler = StandardScaler()
    X_scaled_predict = scaler.fit_transform(X)
    ma_ligne = X_scaled_predict[-1]

    #JOUER LA PREDICTION
    caracteristiques = [ma_ligne]

    # Prévoir la classe avec le modèle
    modele = charger_modele()
    prediction_MLville = modele.predict(caracteristiques)
    st.write('##### Prédiction avec modèle de ML le plus performant (mais caractéristiques de la ville)')
    st.write("Le salaire est",round(prediction_MLville[0],2),"€/h soit environ",round((prediction_MLville[0]*4.5*35),2),"€/mois")
    st.slider('', min_value=6.0, max_value=60.0, value=prediction_MLville[0], step=1.0)

    #2ème modèle
    df=pd.read_csv("df_full_full_catpos.csv")
    df.drop_duplicates(inplace=True)
    df_debase = df
    df_debase = df_debase.drop(['nom_DEP','nom_REG','salaire_moyen'],axis=1)
    df=df.drop(['CODGEO', 'LIBGEO',"nom_REG", 'nom_DEP'],axis=1)

    X = df.drop('salaire_moyen', axis=1) 
    y = df['salaire_moyen']
    def charger_modele():
        # Charger le modèle à partir du fichier Pickle
        with open('modele.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele

    X=df.drop('salaire_moyen', axis=1) 
    ma_ligne = df_debase.loc[df_debase["LIBGEO"]==commune].iloc[[0]]

    if genre == "H" :
        ma_ligne["genre"]=1
    if genre == "F" :
        ma_ligne["genre"]=2

    if cat_poste == "cadre sup":
        ma_ligne["cat_poste"]=4
    if cat_poste == "cadre":
        ma_ligne["cat_poste"]=3
    if cat_poste == "travailleur":
        ma_ligne["cat_poste"]=2
    if cat_poste == "ouvrier":
        ma_ligne["cat_poste"]=1
    
    ma_ligne=ma_ligne.drop(["LIBGEO","CODGEO"],axis=1)    
    X = pd.concat([X, ma_ligne], ignore_index=True)

    scaler = StandardScaler()
    X_scaled_predict = scaler.fit_transform(X)
    ma_ligne = X_scaled_predict[-1]

    #JOUER LA PREDICTION
    caracteristiques = [ma_ligne]

    # Prévoir la classe avec le modèle
    modele = charger_modele()
    prediction_MLind = modele.predict(caracteristiques)
    st.write('##### Prédiction avec modèle de ML caractéristiques de la personne')
    st.write("Le salaire est",round(prediction_MLind[0],2),"€/h soit environ",round((prediction_MLind[0]*4.5*35),2),"€/mois")
    st.slider('', min_value=6.0, max_value=60.0, value=prediction_MLind[0], step=1.0)

    #Prédiction moulinette
    dfsal = pd.read_csv("net_salary_per_town_categories.csv")

    def devine_salaire(genre, ville, age, profession,dep="") :
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
    
    if cat_poste == "cadre sup":
        cat_poste=4
    if cat_poste == "cadre":
        cat_poste=3
    if cat_poste == "travailleur":
        cat_poste=2
    if cat_poste == "ouvrier":
        cat_poste=1
    
    #prédiction
    prediction_moyenne_fonction=round(devine_salaire(genre, commune, age, cat_poste),2)
    st.write('##### Prédiction avec une fonction')
    st.write("Le salaire est",prediction_moyenne_fonction,"€/h soit environ",round((prediction_moyenne_fonction*4.5*35),2),"€/mois")
    st.slider('', min_value=6.0, max_value=60.0, value=prediction_moyenne_fonction, step=1.0)


if page == pages[5] :
    st.header("📌 Conclusion")
    st.markdown("""<style>h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)
    
    st.subheader("Comment s'explique les différences de salaire en France ?")
    st.write("L'Île de France concentre le plus grand nombre d'entreprises, la plus grande densité mais aussi la plus grande proportion de cadre. C'est cette variable qui semble tirer les salaires vers le haut.")
    st.write("Sur un marché de l'emploi tendu, on peut logiquement comprendre que les salaires des cadres seront donc vus à la hausse pour attirer les talents.")

    st.subheader("Comment gagner un salaire haut en France ?")

    with st.expander("Réponse 1"):
        st.write("<div style='text-align: center;font-size:50px;'>🏙️ 🚇 🏛️ 🗼</div>", unsafe_allow_html=True)
        st.write("En habitant dans une ville avec une grande proportion de cadre, idéalement en Île de France")

    with st.expander("Réponse 2"):
        st.write("<div style='text-align: center;font-size:50px;'>💼 👔 📊 💻</div>", unsafe_allow_html=True)
        st.write("En étant soi-même cadre supérieur ou cadre")

    with st.expander("Réponse 3"):
        st.write("<div style='text-align: center;font-size:50px;'>👨 🕺 💰 🤑</div>", unsafe_allow_html=True)        
        st.write("En étant un homme. A niveau de travail égal, l'homme gagne plus.")

    if st.button("C'est terminé"):
        st.image('question.gif')
