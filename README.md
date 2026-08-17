# Credit scoring Home Credit

Projet portfolio sur le scoring d'octroi de credit a partir du dataset [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

Le projet compare deux approches sur le meme split :

- une scorecard interpretable basee sur une regression logistique, avec score 300-850 ;
- un modele LightGBM oriente performance.

Les metriques exposees sont l'AUC-ROC, le Gini, le KS, l'average precision, le Brier score et le PSI train/test. Le dashboard affiche aussi la sensibilite des decisions selon plusieurs seuils de probabilite.

## Demarrage rapide

```bash
uv sync --extra dev
uv run streamlit run app/streamlit_app.py
```

Puis ouvrir http://localhost:8501. Dans la barre laterale, choisir la taille de l'echantillon et cliquer sur **Entraîner les modèles**.

## Entrainement en ligne de commande

```bash
uv run credit-scoring --sample-size 25000
uv run credit-scoring --sample-size 25000 --include-related
```

Les modeles et les metriques sont ecrits dans `models/`. Les tables relationnelles sont optionnelles car elles augmentent fortement le temps et la memoire necessaires.

La scorecard accepte des parametres metier explicites (`base_score`, `base_odds` et `points_to_double_odds`) pour adapter la conversion de probabilite en score 300-850. Le modele LightGBM n'applique plus automatiquement une ponderation de classes : ses probabilites restent ainsi alignees sur la prevalence observee, sous reserve d'une calibration ulterieure.

## Tests

```bash
uv run pytest
```

## Organisation

```text
src/credit_scoring/
├── data/         # chargement et agregations relationnelles
├── scorecard/    # regression logistique et conversion en score
├── boosting/     # LightGBM
├── evaluation/  # AUC, Gini, KS, PSI
└── explain/      # utilitaires d'explication
app/              # dashboard Streamlit
tests/            # tests unitaires
```

Le dataset brut est volontairement conserve dans `data/raw/` mais les artefacts d'entrainement et les donnees transformees ne doivent pas etre commites.
