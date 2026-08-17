"""Dashboard Streamlit de simulation de credit scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_scoring.boosting.train import BoostingModel
from credit_scoring.data.build_features import build_features, split_target
from credit_scoring.data.load import DEFAULT_RAW_DIR, load_application
from credit_scoring.evaluation.metrics import evaluate_predictions, threshold_table
from credit_scoring.scorecard.model import ScorecardModel


st.set_page_config(
    page_title="Credit scoring | Home Credit",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    """Charge la table principale une seule fois par session serveur."""
    return load_application(raw_dir=DEFAULT_RAW_DIR)


@st.cache_resource(show_spinner="Entraînement des modèles en cours…")
def train_cached(sample_size: int, include_related: bool) -> tuple[ScorecardModel, BoostingModel, dict]:
    """Entraine et conserve les modeles selon les choix de l'utilisateur."""
    application = load_training_data()
    if sample_size < len(application):
        application = application.sample(sample_size, random_state=42)
    features = build_features(application, raw_dir=DEFAULT_RAW_DIR, include_related=include_related)
    x, y = split_target(features)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    scorecard = ScorecardModel().fit(x_train, y_train)
    boosting = BoostingModel().fit(x_train, y_train)
    scorecard_train_probability = scorecard.predict_proba(x_train)
    scorecard_probability = scorecard.predict_proba(x_test)
    boosting_train_probability = boosting.predict_proba(x_train)
    boosting_probability = boosting.predict_proba(x_test)
    metrics = {
        "scorecard": evaluate_predictions(
            y_test,
            scorecard_probability,
            reference_probability=scorecard_train_probability,
        ),
        "boosting": evaluate_predictions(
            y_test,
            boosting_probability,
            reference_probability=boosting_train_probability,
        ),
        "rows": len(features),
        "default_rate": float(y.mean()),
        "policy": {
            "scorecard": threshold_table(y_test, scorecard_probability).to_dict("records"),
            "boosting": threshold_table(y_test, boosting_probability).to_dict("records"),
        },
    }
    return scorecard, boosting, metrics


def make_prediction_row(model: ScorecardModel | BoostingModel, values: dict[str, object]) -> pd.DataFrame:
    """Construit une observation compatible avec les colonnes d'entrainement."""
    if not model.feature_columns:
        raise RuntimeError("Le modele n'a pas de colonnes memorisees.")
    row = pd.DataFrame({column: [None] for column in model.feature_columns})
    for column, value in values.items():
        if column in row.columns:
            row.loc[0, column] = value
    return row


def risk_label(probability: float) -> tuple[str, str]:
    """Retourne un libelle et une couleur pour la probabilité de defaut."""
    if probability < 0.10:
        return "Risque faible", "#17805c"
    if probability < 0.25:
        return "Risque modéré", "#b7791f"
    return "Risque élevé", "#c53030"


st.title("Credit scoring Home Credit")
st.caption("Comparer une scorecard interprétable et un modèle LightGBM sur le risque de défaut.")

with st.sidebar:
    st.header("Configuration")
    with st.form("training_form"):
        sample_size = st.selectbox(
            "Lignes d'entraînement",
            options=[5_000, 10_000, 25_000, 50_000, 100_000, 307_511],
            index=2,
            format_func=lambda value: f"{value:,}".replace(",", " "),
        )
        include_related = st.toggle(
            "Ajouter les tables relationnelles",
            value=False,
            help="Plus riche, mais plus long et plus gourmand en mémoire.",
        )
        run_training = st.form_submit_button("Entraîner les modèles", type="primary", width="stretch")

    if "models" not in st.session_state:
        st.info("Choisis une taille puis lance un premier entraînement.")

if run_training or "models" not in st.session_state:
    if run_training:
        scorecard, boosting, metrics = train_cached(sample_size, include_related)
        st.session_state.models = (scorecard, boosting)
        st.session_state.metrics = metrics
        st.session_state.config = (sample_size, include_related)

if "models" in st.session_state:
    scorecard, boosting = st.session_state.models
    metrics = st.session_state.metrics
    scorecard_metrics = metrics["scorecard"]
    boosting_metrics = metrics["boosting"]

    overview, simulation = st.tabs(["Vue générale", "Simulation de décision"])
    with overview:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Observations", f"{metrics['rows']:,}".replace(",", " "))
        kpi2.metric("Taux de défaut", f"{metrics['default_rate']:.1%}")
        kpi3.metric("AUC scorecard", f"{scorecard_metrics['auc_roc']:.3f}")
        kpi4.metric("AUC LightGBM", f"{boosting_metrics['auc_roc']:.3f}")

        st.subheader("Comparaison des performances")
        comparison = pd.DataFrame(
            {
                "Métrique": ["AUC-ROC", "Gini", "KS", "Average precision", "Brier", "PSI train/test"],
                "Scorecard": [
                    scorecard_metrics["auc_roc"],
                    scorecard_metrics["gini"],
                    scorecard_metrics["ks"],
                    scorecard_metrics["average_precision"],
                    scorecard_metrics["brier_score"],
                    scorecard_metrics.get("psi", float("nan")),
                ],
                "LightGBM": [
                    boosting_metrics["auc_roc"],
                    boosting_metrics["gini"],
                    boosting_metrics["ks"],
                    boosting_metrics["average_precision"],
                    boosting_metrics["brier_score"],
                    boosting_metrics.get("psi", float("nan")),
                ],
            }
        ).set_index("Métrique")
        st.bar_chart(comparison, y_label="Valeur", x_label="Métrique")
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")

        st.subheader("Sensibilité de la décision")
        policy_records = metrics.get("policy", {}).get("scorecard", [])
        if policy_records:
            policy = pd.DataFrame(policy_records)
            st.dataframe(policy.style.format("{:.3f}"), width="stretch")
        else:
            st.caption("Relance l'entraînement pour calculer la sensibilité des seuils.")

        st.subheader("Variables dominantes de la scorecard")
        importance = scorecard.feature_importance(top_n=15).set_index("feature")
        st.bar_chart(importance["importance"], horizontal=True, x_label="Importance absolue")

    with simulation:
        st.subheader("Simuler une demande")
        st.caption("Les champs non affichés restent imputés selon les données d'entraînement.")
        with st.form("simulation_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                income = st.number_input("Revenu annuel", min_value=10_000.0, value=180_000.0, step=10_000.0)
                credit = st.number_input("Montant du crédit", min_value=10_000.0, value=500_000.0, step=25_000.0)
                annuity = st.number_input("Annuité", min_value=1_000.0, value=25_000.0, step=1_000.0)
            with col2:
                age = st.number_input("Âge", min_value=18, max_value=90, value=35)
                employment = st.number_input("Années d'emploi", min_value=0, max_value=60, value=5)
                children = st.number_input("Nombre d'enfants", min_value=0, max_value=20, value=0)
            with col3:
                gender = st.selectbox("Genre", ["F", "M"])
                education = st.selectbox(
                    "Niveau d'éducation",
                    ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"],
                )
                income_type = st.selectbox(
                    "Type de revenu",
                    ["Working", "Commercial associate", "Pensioner", "State servant"],
                )
            decision_threshold = st.slider(
                "Seuil d'acceptation (scorecard)",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.01,
                help="Une demande est acceptée si la probabilité estimée reste sous ce seuil.",
            )
            simulate = st.form_submit_button("Évaluer la demande", type="primary", width="stretch")

        if simulate:
            values = {
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "DAYS_BIRTH": -age * 365,
                "DAYS_EMPLOYED": -employment * 365,
                "CNT_CHILDREN": children,
                "CODE_GENDER": gender,
                "NAME_EDUCATION_TYPE": education,
                "NAME_INCOME_TYPE": income_type,
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
            }
            scorecard_row = make_prediction_row(scorecard, values)
            boosting_row = make_prediction_row(boosting, values)
            scorecard_probability = float(scorecard.predict_proba(scorecard_row)[0])
            boosting_probability = float(boosting.predict_proba(boosting_row)[0])
            score = float(scorecard.score(scorecard_row)[0])
            label, color = risk_label(scorecard_probability)

            st.divider()
            result1, result2, result3 = st.columns(3)
            result1.metric("Probabilité scorecard", f"{scorecard_probability:.1%}")
            result2.metric("Probabilité LightGBM", f"{boosting_probability:.1%}")
            result3.metric("Score 300-850", f"{score:.0f}")
            st.markdown(f"### :{('green' if color == '#17805c' else 'orange' if color == '#b7791f' else 'red')}[{label}]")
            st.progress(min(scorecard_probability, 1.0), text="Probabilité estimée de défaut")
            decision = "Accepter" if scorecard_probability < decision_threshold else "Revoir / refuser"
            st.info(f"Décision indicative : **{decision}** (seuil {decision_threshold:.0%}).")
            st.subheader("Facteurs principaux de la scorecard")
            st.dataframe(scorecard.explain(scorecard_row, top_n=8), width="stretch")
else:
    st.info("Lance un entraînement depuis la barre latérale pour afficher le dashboard.")
