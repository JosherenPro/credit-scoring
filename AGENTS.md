# AGENT.md — Système de Scoring d'Octroi de Crédit

## Contexte du projet

Projet portfolio personnel (type Kaggle) sur le **scoring d'octroi de crédit**, avec un double objectif :
comparer une approche **interprétable/réglementaire** (scorecard classique WOE/IV + régression logistique)
à une approche **performance** (gradient boosting moderne), sur un même dataset.

Dataset : **Home Credit Default Risk** (Kaggle) — données relationnelles multi-tables
(`application_train/test`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`,
`installments_payments`, `credit_card_balance`). Target binaire déséquilibrée (~8% de défaut).

## Objectifs

1. Construire un pipeline de feature engineering robuste sur les tables relationnelles.
2. Développer une branche **scorecard interprétable** : binning, WOE, IV, régression logistique, score type 300-850.
3. Développer une branche **performance** : LightGBM/XGBoost + Optuna + SHAP.
4. Comparer les deux branches sur des métriques standards du domaine (AUC-ROC, KS, Gini, PSI).
5. (Bonus) Dashboard Streamlit de simulation de décision de crédit avec explication SHAP.

## Stack technique

- Gestionnaire d'environnement : **uv** (pas de pip/conda direct)
- Structure projet : **src-layout**
- Manipulation de données : `pandas` / `polars` (tables volumineuses)
- ML performance : `lightgbm`, `xgboost`, `optuna`
- ML interprétable : `scikit-learn` (LogisticRegression), `scorecardpy`
- Interprétabilité : `shap`
- Visualisation : `matplotlib`, `seaborn`
- Dashboard (bonus) : `streamlit`
- Notebooks : Jupyter, un notebook par grande étape (pas un notebook monolithique)

## Structure du projet

```
credit-scoring/
├── pyproject.toml
├── AGENT.md
├── README.md
├── data/
│   ├── raw/                  # données brutes Home Credit (non versionnées)
│   ├── interim/               # tables jointes/agrégées intermédiaires
│   └── processed/              # features finales prêtes pour modélisation
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_scorecard_woe.ipynb
│   ├── 04_lightgbm_optuna.ipynb
│   └── 05_comparatif_final.ipynb
├── src/
│   └── credit_scoring/
│       ├── __init__.py
│       ├── data/
│       │   ├── load.py         # chargement des tables brutes
│       │   └── build_features.py  # agrégations bureau/previous_application etc.
│       ├── scorecard/
│       │   ├── binning.py      # binning + calcul WOE/IV
│       │   └── model.py        # régression logistique + scoring 300-850
│       ├── boosting/
│       │   ├── train.py        # entraînement LightGBM/XGBoost
│       │   └── tuning.py       # optimisation Optuna
│       ├── evaluation/
│       │   └── metrics.py      # AUC, KS, Gini, PSI
│       └── explain/
│           └── shap_utils.py
├── app/
│   └── streamlit_app.py        # dashboard bonus
└── tests/
    └── ...
```

## Conventions de code

- Python 3.11+, typage (`from __future__ import annotations`, type hints partout).
- Fonctions courtes, docstrings en français, noms de variables en anglais (convention data science).
- Pas de notebook comme source de vérité : toute logique réutilisable part dans `src/credit_scoring/`,
  les notebooks ne font qu'appeler ces fonctions et afficher des résultats.
- Commits atomiques, messages en français, préfixés (`feat:`, `fix:`, `data:`, `exp:`).
- Aucune donnée brute ni artefact volumineux commité (`data/raw/`, `data/interim/` dans `.gitignore`).

## Commandes utiles

```bash
uv sync                                   # installer les dépendances
uv run jupyter lab                        # lancer les notebooks
uv run python -m credit_scoring.data.load # charger/valider les données brutes
uv run streamlit run app/streamlit_app.py # lancer le dashboard bonus
uv run pytest                             # lancer les tests
```

## Métriques d'évaluation attendues

- **AUC-ROC** et **Gini** (= 2·AUC − 1) : discrimination globale.
- **KS statistic** : séparation max entre distributions bons/mauvais payeurs (standard métier crédit).
- **PSI** (Population Stability Index) : stabilité du score si split temporel simulé.
- **IV (Information Value)** par variable dans la branche scorecard.
- Temps d'inférence et taille du modèle comme critère secondaire du comparatif.

## Points d'attention spécifiques à Home Credit

- Le `missingness` (valeurs manquantes) est souvent informatif — ne pas imputer aveuglément avant
  d'avoir vérifié si l'absence de donnée corrèle avec la target.
- Bien séparer les agrégations par `SK_ID_CURR` pour éviter les fuites de données (data leakage) entre
  `previous_application` / `bureau` et la table principale.
- Attention au déséquilibre de classes : ne pas se fier à l'accuracy, toujours reporter AUC/KS/Gini.
- Garder une trace claire de quelles variables viennent de quelle table source (préfixes de colonnes).

## Ce que l'agent NE doit PAS faire

- Ne pas fabriquer de résultats de métriques sans exécution réelle du code.
- Ne pas mélanger la logique des deux branches (scorecard vs boosting) dans les mêmes modules.
- Ne pas committer de données brutes ou de modèles binaires volumineux.
- Ne pas utiliser pip/conda directement — toujours passer par `uv`.