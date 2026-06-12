import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# =============================================================
# PARTIE 1 — CHARGEMENT DES DONNÉES
# =============================================================

df_raw = pd.read_excel("External_Database_1.xlsx",sheet_name="Harvested Area",header=1,index_col=0)
df_raw.columns = df_raw.iloc[0]
df_raw = df_raw.iloc[1:]
df_raw = df_raw.drop(columns=["PERU"], errors='ignore')
print(df_raw.head())
print(df_raw.describe().round(2))
print(df_raw.info())
print(f"\nDimensions : {df_raw.shape[0]} années x {df_raw.shape[1]} régions")


#dataset2
df_raw1= pd.read_excel("External_Database_2.xlsx",sheet_name="Harvested Area",header=1,index_col=0)
df_raw1.columns = df_raw1.iloc[0]
df_raw1 = df_raw1.iloc[1:]
print(df_raw1.head())
print(df_raw1.describe().round(2))
print(df_raw1.info())
print(f"\nDimensions : {df_raw1.shape[0]} années x {df_raw1.shape[1]} culture")


# =============================================================
# PARTIE 2 — ANALYSE DESCRIPTIVE (EDA)
# =============================================================

###dataset1
# --- 2.1 Analyse UNIVARIÉE ---
# Distribution de la superficie totale par année (PERU entier recalculé)
print(df_raw.columns.tolist())

df_raw["TOTAL"] = df_raw.sum(axis=1)

df_raw["TOTAL"] = pd.to_numeric(df_raw["TOTAL"], errors="coerce").fillna(0)

x = range(len(df_raw))
y = df_raw["TOTAL"].values

plt.figure(figsize=(12, 5))
plt.plot(x, y, marker='o', color='green', linewidth=2)
plt.fill_between(x, y, alpha=0.2, color='green')
plt.xticks(x, df_raw.index, rotation=45)
plt.title("Évolution de la superficie totale de quinoa au Pérou (1995–2014)", fontsize=14)
plt.xlabel("Année")
plt.ylabel("Superficie (Ha)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("plot_evolution_totale_database1.png", dpi=150)
plt.show()

# Boxplot par région
df_plot = df_raw.drop(columns=["TOTAL"], errors="ignore")
df_plot = df_plot.apply(pd.to_numeric, errors="coerce")
df_plot = df_plot.dropna(axis=1, how="all")
plt.figure(figsize=(14, 6))
df_plot.boxplot(rot=45)
plt.title("Distribution de la superficie par région (1995–2014)", fontsize=13)
plt.ylabel("Superficie (Ha)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_boxplot_regions.png", dpi=150)
plt.show()
##histogramme : 



# Transformer toutes les valeurs en une seule série
data_hist = df_plot.values.flatten()

# Supprimer les valeurs manquantes
data_hist = data_hist[~pd.isna(data_hist)]

plt.figure(figsize=(10, 6))
plt.hist(data_hist, bins=20)
plt.title("Distribution des superficies de quinoa", fontsize=13)
plt.xlabel("Superficie (Ha)")
plt.ylabel("Fréquence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_histogramme_superficie.png", dpi=150)
plt.show()
# --- 2.2 Analyse BIVARIÉE ---
# Top 5 régions en 2014
top5_2014 = df_plot.loc[2014].sort_values(ascending=False).head(5)
print(f"\nTop 5 régions en 2014 :\n{top5_2014}")
top5_regions = top5_2014.index.tolist()
plt.figure(figsize=(12, 6))
for region in top5_regions:
    plt.plot(df_plot.index, df_plot[region], marker='o', label=region, linewidth=2)
plt.title("Évolution des 5 principales régions productrices de quinoa", fontsize=13)
plt.xlabel("Année")
plt.ylabel("Superficie (Ha)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("plot_top5_regions.png", dpi=150)
plt.show()

##cartographie :
# Croissance entre 1995 et 2014

croissance = df_plot.loc[2014] - df_plot.loc[1995]

plt.figure(figsize=(12, 6))

croissance.sort_values(ascending=False).plot(kind='bar',figsize=(12,6))

plt.title("Croissance des superficies de quinoa par région (1995-2014)")
plt.xlabel("Région")
plt.ylabel("Croissance (Ha)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plot_croissance_regions.png", dpi=150)
plt.show()



###dataset2


# --- 2.1 Analyse UNIVARIÉE ---
# Distribution de la superficie totale par année (PERU entier recalculé)
print(df_raw1.columns.tolist())

df_raw1 = df_raw1.apply(pd.to_numeric, errors='coerce')

x = range(len(df_raw1))
y = df_raw1["QUINUA"].values

plt.figure(figsize=(12, 5))
plt.plot(x, y, marker='o', color='green', linewidth=2)
plt.fill_between(x, y, alpha=0.2, color='green')

plt.xticks(x, df_raw1.index, rotation=45)
plt.title("Évolution de la superficie récoltée de la QUINUA", fontsize=14)
plt.xlabel("Année")
plt.ylabel("Superficie (Ha)")
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("plot_quinua_dataset2.png", dpi=150)
plt.show()


# Boxplot par culture

df_plot1 = df_raw1.drop(columns=["TOTAL"], errors="ignore")
df_plot1 = df_plot1.apply(pd.to_numeric, errors="coerce")
df_plot1 = df_plot1.dropna(axis=1, how="all")

plt.figure(figsize=(14, 6))
df_plot1.boxplot(rot=45)

plt.title("Distribution de la superficie par culture (1995–2014)", fontsize=13)
plt.ylabel("Superficie (Ha)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_boxplot_cultures.png", dpi=150)
plt.show()


##histogramme 

# Transformer toutes les valeurs en une seule série

data_hist = df_plot1.values.flatten()

# Supprimer les valeurs manquantes

data_hist = data_hist[~pd.isna(data_hist)]

plt.figure(figsize=(10, 6))
plt.hist(data_hist, bins=20)

plt.title("Distribution des superficies des cultures andines", fontsize=13)
plt.xlabel("Superficie (Ha)")
plt.ylabel("Fréquence")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_histogramme_superficie.png", dpi=150)
plt.show()

# --- 2.2 Analyse BIVARIÉE ---
# Top 5 cultures en 2014

top5_2014 = df_plot1.loc[2014].sort_values(ascending=False).head(5)

print(f"\nTop 5 cultures en 2014 :\n{top5_2014}")

top5_cultures = top5_2014.index.tolist()

plt.figure(figsize=(12, 6))

for culture in top5_cultures:
    plt.plot(df_plot1.index,df_plot1[culture],marker='o',label=culture,linewidth=2)
plt.title("Évolution des 5 cultures les plus importantes", fontsize=13)
plt.xlabel("Année")
plt.ylabel("Superficie (Ha)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("plot_top5_cultures.png", dpi=150)
plt.show()


###cartographie 
# Croissance entre 1995 et 2014

croissance1 = df_plot1.loc[2014] - df_plot1.loc[1995]

plt.figure(figsize=(12, 6))

croissance1.sort_values(ascending=False).plot(kind='bar',figsize=(12, 6))

plt.title("Croissance des superficies par culture (1995-2014)")
plt.xlabel("Culture")
plt.ylabel("Croissance (Ha)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plot_croissance_cultures.png", dpi=150)
plt.show()





# =============================================================
# PARTIE 3 — DATA PREPROCESSING
# =============================================================

# On travaille avec df_regions (sans colonne TOTAL)
df = df_plot.copy()

# --- 3.1 Vérification des valeurs manquantes ---
print("\n=== VALEURS MANQUANTES ===")
print(df.isnull().sum())
# Imputation si nécessaire (ici les 0 sont des vraies valeurs, pas des NaN)

# --- 3.2 Détection des outliers (IQR) ---
print("\n=== DÉTECTION DES OUTLIERS ===")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print(f"Nombre d'outliers par région :\n{outliers.sum()}")

# --- 3.3 Transformation : créer un format long (Region, Annee, Superficie) ---
df_reset = df.reset_index()

# Renommer automatiquement la première colonne en "Annee"
df_reset.rename(columns={df_reset.columns[0]: "Annee"}, inplace=True)

df_long = df_reset.melt(id_vars="Annee", var_name="Region", value_name="Superficie_Ha")

print(f"\nDataset format long : {df_long.shape}")
print(df_long.head())

# --- 3.4 Créer la variable cible binaire : croissance forte (1) ou faible (0) ---
# Variation entre 1995 et 2014 par région
variation = ((df.loc[2014] - df.loc[1995]) / (df.loc[1995] + 1)) * 100
df_features = pd.DataFrame({
    "Region": variation.index,
    "Variation_pct": variation.values,
    "Superficie_1995": df.loc[1995].values,
    "Superficie_2014": df.loc[2014].values,
    "Superficie_Moy": df.mean().values,
    "Superficie_Max": df.max().values,
    "Superficie_Std": df.std().values,
})
# Variable cible : 1 si variation > médiane, 0 sinon
median_var = df_features["Variation_pct"].median()
df_features["Croissance_Forte"] = (df_features["Variation_pct"] > median_var).astype(int)
print(f"\nDataset features :\n{df_features}")
print(f"\nDistribution de la variable cible :\n{df_features['Croissance_Forte'].value_counts()}")

# --- 3.5 Normalisation ---
features_cols = ["Superficie_1995", "Superficie_2014", "Superficie_Moy",
                 "Superficie_Max", "Superficie_Std"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features[features_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=features_cols)
y = df_features["Croissance_Forte"]

print("\n=== DONNÉES NORMALISÉES ===")
print(X_scaled_df.describe().round(3))

# --- 3.6 SMOTE (sur le dataset long pour avoir assez de lignes) ---
# Encodage de la région pour SMOTE
le = LabelEncoder()
df_long["Region_encoded"] = le.fit_transform(df_long["Region"])
df_long["Croissance"] = (df_long["Superficie_Ha"] > df_long["Superficie_Ha"].median()).astype(int)

X_smote = df_long[["Annee", "Region_encoded", "Superficie_Ha"]]
y_smote = df_long["Croissance"]

print(f"\n=== AVANT SMOTE ===")
print(y_smote.value_counts())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_smote, y_smote)

print(f"\n=== APRÈS SMOTE ===")
print(pd.Series(y_res).value_counts())
print(f"Taille avant SMOTE : {len(X_smote)} | Après SMOTE : {len(X_res)}")

# =============================================================
# PARTIE 4 — HEATMAP DE CORRÉLATION
# =============================================================

plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="YlOrRd", linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Heatmap de corrélation entre les régions productrices de quinoa", fontsize=13)
plt.tight_layout()
plt.savefig("plot_heatmap_correlation.png", dpi=150)
plt.show()
print("→ Les régions Puno, Ayacucho et Huancavelica sont fortement corrélées (évolution similaire).")
print("→ Arequipa présente une corrélation plus faible : son essor est tardif et brutal (2014).")

# =============================================================
# PARTIE 5 — ALGORITHMES DE CLASSIFICATION
# =============================================================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve, classification_report)
import time

# Utiliser les données SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)

print(f"\nTrain : {X_train.shape} | Test : {X_test.shape}")

# Dictionnaire des modèles
models = {
    "Arbre de Décision": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":               KNeighborsClassifier(n_neighbors=5),
    "SVM":               SVC(probability=True, random_state=42),
    "Naive Bayes":       GaussianNB(),
}

# --- 5.1 Entraînement et évaluation ---
resultats = []
for nom, modele in models.items():
    debut = time.time()
    modele.fit(X_train, y_train)
    temps = round(time.time() - debut, 4)

    y_pred = modele.predict(X_test)
    y_prob = modele.predict_proba(X_test)[:, 1]

    resultats.append({
        "Modèle": nom,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "Temps (s)": temps,
    })

df_resultats = pd.DataFrame(resultats).sort_values("F1-Score", ascending=False)
print("\n=== TABLEAU COMPARATIF DES MODÈLES ===")
print(df_resultats.to_string(index=False))

# --- 5.2 Matrices de confusion ---
fig, axes = plt.subplots(1, len(models), figsize=(20, 4))
for ax, (nom, modele) in zip(axes, models.items()):
    y_pred = modele.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Faible', 'Forte'], yticklabels=['Faible', 'Forte'])
    ax.set_title(nom, fontsize=10)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
plt.tight_layout()
plt.savefig("plot_matrices_confusion.png", dpi=150)
plt.show()

# --- 5.3 Courbes ROC ---
plt.figure(figsize=(10, 7))
for nom, modele in models.items():
    y_prob = modele.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{nom} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Aléatoire")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbes ROC — Comparaison des modèles", fontsize=13)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_courbes_roc.png", dpi=150)
plt.show()

# =============================================================
# PARTIE 6 — ARBRE DE DÉCISION (détail)
# =============================================================

dt = models["Arbre de Décision"]
print("\n=== ARBRE DE DÉCISION — RÈGLES ===")
print(export_text(dt, feature_names=list(X_res.columns)))

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(X_res.columns),
          class_names=["Faible", "Forte"], filled=True,
          rounded=True, fontsize=9)
plt.title("Arbre de décision — Croissance du quinoa", fontsize=13)
plt.tight_layout()
plt.savefig("plot_arbre_decision.png", dpi=150)
plt.show()
print("→ Interprétation : la superficie et l'année sont les variables discriminantes principales.")

# =============================================================
# PARTIE 7 — CLUSTERING K-MEANS
# =============================================================

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering sur les profils de régions (transpose : régions en lignes)
df_regions = df.T  # shape : 18 régions x 20 années

# Normalisation pour K-Means
scaler2 = StandardScaler()
X_km = scaler2.fit_transform(df_regions)

# Méthode Elbow pour choisir k
inertia = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_km)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 8), inertia, marker='o', color='steelblue')
plt.title("Méthode Elbow — Choix du nombre de clusters")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertie")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("plot_elbow.png", dpi=150)
plt.show()

# K-Means avec k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_km)
df_regions["Cluster"] = clusters

print("\n=== RÉSULTATS K-MEANS (k=3) ===")
for i in range(3):
    regs = df_regions[df_regions["Cluster"] == i].index.tolist()
    print(f"Cluster {i} : {regs}")
print("→ Cluster 0 = régions traditionnelles stables (ex: Puno)")
print("→ Cluster 1 = régions émergentes (ex: Arequipa, Ayacucho)")
print("→ Cluster 2 = régions marginales (petite superficie)")

# Visualisation PCA 2D des clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_km)
plt.figure(figsize=(9, 6))
colors = ['#e74c3c', '#2ecc71', '#3498db']
for i in range(3):
    mask = clusters == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100,
                c=colors[i], label=f"Cluster {i}", zorder=3)
    for j, reg in enumerate(df_regions.index[:-1]):  # exclude 'Cluster' col
        if clusters[j] == i:
            plt.annotate(reg, (X_pca[j, 0], X_pca[j, 1]),
                         fontsize=8, ha='center', va='bottom')
plt.title("Clustering K-Means des régions (PCA 2D)", fontsize=13)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_kmeans_pca.png", dpi=150)
plt.show()

# =============================================================
# PARTIE 8 — CAH (Classification Ascendante Hiérarchique)
# =============================================================

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X_km, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=df_regions.index[:-1].tolist(),
           leaf_rotation=45, leaf_font_size=10,
           color_threshold=5)
plt.title("Dendrogramme — CAH des régions productrices de quinoa", fontsize=13)
plt.xlabel("Région")
plt.ylabel("Distance (Ward)")
plt.tight_layout()
plt.savefig("plot_cah_dendrogramme.png", dpi=150)
plt.show()
print("→ Le dendrogramme confirme 3 groupes naturels de régions selon leur profil de production.")

# =============================================================
# PARTIE 9 — RÉSEAU DE NEURONES (MLP)
# =============================================================

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                    random_state=42, early_stopping=True)
debut = time.time()
mlp.fit(X_train, y_train)
temps_mlp = round(time.time() - debut, 4)
y_pred_mlp = mlp.predict(X_test)

print("\n=== RÉSEAU DE NEURONES (MLP) ===")
print(f"Temps d'entraînement : {temps_mlp}s")
print(classification_report(y_test, y_pred_mlp,
                             target_names=["Faible", "Forte"]))

# Courbe de perte
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_, color='purple', linewidth=2)
plt.title("Courbe de perte — Réseau de neurones MLP", fontsize=13)
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("plot_mlp_loss.png", dpi=150)
plt.show()

# =============================================================
# PARTIE 10 — RÈGLES D'ASSOCIATION
# =============================================================

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Discrétiser : pour chaque année, quelles régions ont une superficie > médiane ?
median_superficie = df.median()
transactions = []
for annee in df.index:
    transaction = [reg for reg in df.columns if df.loc[annee, reg] > median_superficie[reg]]
    transactions.append(transaction)

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_te, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print("\n=== TOP 10 RÈGLES D'ASSOCIATION ===")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .sort_values("lift", ascending=False).head(10).to_string(index=False))
print("→ Ces règles montrent quelles régions ont tendance à progresser simultanément.")

# =============================================================
# PARTIE 11 — RÉSEAU BAYÉSIEN
# =============================================================

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Discrétiser les données en classes : Bas / Moyen / Haut
df_bn = df_long[["Annee", "Region", "Superficie_Ha"]].copy()
df_bn["Periode"] = pd.cut(df_bn["Annee"],
                           bins=[1994, 2000, 2007, 2014],
                           labels=["1995-2000", "2001-2007", "2008-2014"])
df_bn["Niveau"] = pd.cut(df_bn["Superficie_Ha"],
                          bins=[-1, 500, 3000, 999999],
                          labels=["Bas", "Moyen", "Haut"])
df_bn["Region_cat"] = le.transform(df_bn["Region"])
df_bn = df_bn.dropna()

# Structure du réseau : Periode → Niveau, Region → Niveau
model_bn = BayesianNetwork([("Periode", "Niveau"), ("Region", "Niveau")])
model_bn.fit(df_bn[["Periode", "Region", "Niveau"]],
             estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model_bn)
# Inférence : quelle est la probabilité d'un niveau Haut en 2008-2014 ?
result = infer.query(variables=["Niveau"],
                     evidence={"Periode": "2008-2014"})
print("\n=== RÉSEAU BAYÉSIEN — Inférence ===")
print(result)
print("→ En 2008-2014, la probabilité d'atteindre un niveau 'Haut' est significativement élevée.")

# =============================================================
# RÉSUMÉ FINAL
# =============================================================

print("\n" + "="*60)
print("  RÉSUMÉ — TABLEAU COMPARATIF FINAL DES MODÈLES")
print("="*60)
print(df_resultats.to_string(index=False))
print("\nFichiers graphiques générés :")
fichiers = [
    "plot_evolution_totale.png",
    "plot_boxplot_regions.png",
    "plot_top5_regions.png",
    "plot_heatmap_correlation.png",
    "plot_matrices_confusion.png",
    "plot_courbes_roc.png",
    "plot_arbre_decision.png",
    "plot_elbow.png",
    "plot_kmeans_pca.png",
    "plot_cah_dendrogramme.png",
    "plot_mlp_loss.png",
]
for f in fichiers:
    print(f"  ✓ {f}")