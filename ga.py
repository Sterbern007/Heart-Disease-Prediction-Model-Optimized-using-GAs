from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from main import ExperimentConfig, run_experiment, train_prediction_pipeline
from model import MODEL_LABELS, get_default_hyperparameters
from multioutput_ga import MultiOutputGAConfig, predict_multioutput_patient, run_multioutput_feature_ga
from utils import (
    create_generation_figure,
    create_pareto_figure,
    format_percent,
    format_seconds,
    get_dataset_profile,
    load_heart_disease_dataset,
)


st.set_page_config(
    page_title="CardioGA Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #07131a;
  --panel: rgba(16, 31, 40, 0.45);
  --panel-strong: rgba(12, 26, 33, 0.65);
  --border: rgba(124, 181, 199, 0.15);
  --border-hover: rgba(124, 181, 199, 0.35);
  --text: #edf2f4;
  --muted: #a7bcc2;
  --accent: #53d3a6;
  --warm: #f2b35b;
  --alert: #ff7a59;
  --line: #8bd3e6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 10% 20%, rgba(83, 211, 166, 0.15), transparent 40%),
    radial-gradient(circle at 90% 80%, rgba(242, 179, 91, 0.15), transparent 40%),
    radial-gradient(circle at 50% 50%, rgba(139, 211, 230, 0.05), transparent 60%),
    linear-gradient(135deg, #07131a 0%, #0c1c24 50%, #112830 100%);
  color: var(--text);
  background-attachment: fixed;
}

[data-testid="stSidebar"] {
  background: rgba(7, 19, 26, 0.6);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid var(--border);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 3rem;
  border-radius: 32px;
  background: linear-gradient(135deg, rgba(20, 38, 48, 0.65) 0%, rgba(12, 29, 36, 0.45) 100%);
  border: 1px solid var(--border);
  box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  transition: transform 0.4s ease, box-shadow 0.4s ease;
}

.hero:hover {
  transform: translateY(-5px);
  box-shadow: 0 40px 80px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  border-color: var(--border-hover);
}

.hero::before {
  content: "";
  position: absolute;
  top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(circle at center, rgba(83, 211, 166, 0.1) 0%, transparent 60%);
  animation: rotate 20s linear infinite;
  pointer-events: none;
}

@keyframes rotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.eyebrow {
  margin: 0;
  color: var(--warm);
  text-transform: uppercase;
  letter-spacing: 0.2rem;
  font-size: 0.85rem;
  font-weight: 700;
  font-family: 'Outfit', sans-serif;
  text-shadow: 0 0 20px rgba(242, 179, 91, 0.4);
}

.hero h1 {
  margin: 0.5rem 0 1rem 0;
  font-family: 'Outfit', sans-serif;
  font-weight: 700;
  font-size: clamp(2.5rem, 4.5vw, 4rem);
  line-height: 1.1;
  background: linear-gradient(to right, #ffffff, #a7bcc2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero p {
  max-width: 800px;
  color: var(--muted);
  font-size: 1.1rem;
  line-height: 1.7;
  margin-bottom: 1.5rem;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-top: 1.5rem;
}

.badge {
  padding: 0.5rem 1rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text);
  font-size: 0.85rem;
  font-family: 'Outfit', sans-serif;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
}

.badge:hover {
  background: rgba(255, 255, 255, 0.1);
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.metric-card, .info-card {
  border-radius: 24px;
  padding: 1.5rem;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.metric-card:hover, .info-card:hover {
  transform: translateY(-8px) scale(1.02);
  border-color: var(--border-hover);
  box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.metric-card .label {
  font-size: 0.9rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1rem;
  font-weight: 600;
  font-family: 'Outfit', sans-serif;
}

.metric-card .value {
  margin-top: 0.5rem;
  font-size: 2.5rem;
  font-weight: 700;
  font-family: 'Outfit', sans-serif;
  color: var(--text);
  text-shadow: 0 0 30px rgba(255,255,255,0.1);
}

.metric-card .subtext {
  margin-top: 0.5rem;
  color: var(--muted);
  font-size: 0.95rem;
}

.tone-accuracy { box-shadow: inset 0 0 0 1px rgba(83, 211, 166, 0.3), 0 20px 40px rgba(0, 0, 0, 0.2); }
.tone-accuracy .value { color: var(--accent); text-shadow: 0 0 20px rgba(83, 211, 166, 0.4); }
.tone-accuracy:hover { box-shadow: inset 0 0 0 1px rgba(83, 211, 166, 0.6), 0 30px 60px rgba(83, 211, 166, 0.15); }

.tone-time { box-shadow: inset 0 0 0 1px rgba(242, 179, 91, 0.3), 0 20px 40px rgba(0, 0, 0, 0.2); }
.tone-time .value { color: var(--warm); text-shadow: 0 0 20px rgba(242, 179, 91, 0.4); }
.tone-time:hover { box-shadow: inset 0 0 0 1px rgba(242, 179, 91, 0.6), 0 30px 60px rgba(242, 179, 91, 0.15); }

.tone-fitness { box-shadow: inset 0 0 0 1px rgba(139, 211, 230, 0.3), 0 20px 40px rgba(0, 0, 0, 0.2); }
.tone-fitness .value { color: var(--line); text-shadow: 0 0 20px rgba(139, 211, 230, 0.4); }
.tone-fitness:hover { box-shadow: inset 0 0 0 1px rgba(139, 211, 230, 0.6), 0 30px 60px rgba(139, 211, 230, 0.15); }

.section-title {
  font-family: 'Outfit', sans-serif;
  color: var(--text);
  font-size: 1.7rem;
  font-weight: 600;
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  background: linear-gradient(to right, #ffffff, #a7bcc2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.section-copy {
  color: var(--muted);
  margin-bottom: 1.2rem;
  line-height: 1.7;
  font-size: 1.05rem;
}

.info-card h3 {
  font-family: 'Outfit', sans-serif;
  margin: 0 0 0.5rem 0;
  color: var(--text);
  font-size: 1.25rem;
  font-weight: 600;
}

.info-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}

.parameter-box {
  border-radius: 20px;
  padding: 1.2rem;
  background: var(--panel-strong);
  border: 1px solid var(--border);
  backdrop-filter: blur(20px);
  box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 1rem;
  background: rgba(16, 31, 40, 0.4);
  padding: 0.5rem;
  border-radius: 100px;
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 0.6rem 1.5rem;
  background: transparent;
  color: var(--muted);
  border: none;
  font-family: 'Outfit', sans-serif;
  font-weight: 500;
  transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.05);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(83, 211, 166, 0.2), rgba(83, 211, 166, 0.05));
  color: var(--accent) !important;
  box-shadow: 0 4px 15px rgba(83, 211, 166, 0.15), inset 0 0 0 1px rgba(83, 211, 166, 0.4);
}

div[data-testid="stDataFrame"] {
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: 0 15px 35px rgba(0,0,0,0.2);
}

@media (max-width: 900px) {
  .hero { padding: 2rem; }
  .hero h1 { font-size: 2.2rem; }
}
</style>
"""


def render_metric_card(label: str, value: str, subtext: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def infer_risk_band(probability: float) -> tuple[str, str, str]:
    if probability < 0.35:
        return (
            "Lower Estimated Risk",
            "tone-accuracy",
            "The optimized model sees a lower probability of heart disease presence for this profile.",
        )
    if probability < 0.65:
        return (
            "Borderline Risk",
            "tone-fitness",
            "This profile sits near the decision boundary, so clinical review would be especially important.",
        )
    return (
        "Elevated Risk",
        "tone-time",
        "The profile resembles patients with detected heart disease in the training data more strongly.",
    )


@st.cache_resource(show_spinner=False)
def load_prediction_pipeline(model_name: str):
    hyperparameters = get_default_hyperparameters(model_name)
    pipeline = train_prediction_pipeline(model_name=model_name, hyperparameters=hyperparameters)
    return pipeline, hyperparameters


def build_patient_dataframe(patient_values: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame([patient_values])


def get_option_index(options: dict[str, int], value: int) -> int:
    values = list(options.values())
    return values.index(value)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "experiment_result" not in st.session_state:
    st.session_state["experiment_result"] = None
if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None
if "multioutput_result" not in st.session_state:
    st.session_state["multioutput_result"] = None
if "latest_multioutput_prediction" not in st.session_state:
    st.session_state["latest_multioutput_prediction"] = None

dataset_frame = load_heart_disease_dataset()
initial_profile = get_dataset_profile(dataset_frame)
patient_defaults = {
    "age": int(round(dataset_frame["age"].median())),
    "sex": int(dataset_frame["sex"].mode(dropna=True).iat[0]),
    "cp": int(dataset_frame["cp"].mode(dropna=True).iat[0]),
    "trestbps": int(round(dataset_frame["trestbps"].median())),
    "chol": int(round(dataset_frame["chol"].median())),
    "fbs": int(dataset_frame["fbs"].mode(dropna=True).iat[0]),
    "restecg": int(dataset_frame["restecg"].mode(dropna=True).iat[0]),
    "thalach": int(round(dataset_frame["thalach"].median())),
    "exang": int(dataset_frame["exang"].mode(dropna=True).iat[0]),
    "oldpeak": float(round(dataset_frame["oldpeak"].median(), 1)),
    "slope": int(dataset_frame["slope"].mode(dropna=True).iat[0]),
    "ca": int(dataset_frame["ca"].dropna().mode().iat[0]),
    "thal": int(dataset_frame["thal"].dropna().mode().iat[0]),
}
sex_options = {"Female": 0, "Male": 1}
cp_options = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
fbs_options = {"Below or equal to 120 mg/dl": 0, "Above 120 mg/dl": 1}
restecg_options = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
exang_options = {"No exercise-induced angina": 0, "Exercise-induced angina": 1}
slope_options = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
ca_options = {
    "0 major vessels": 0,
    "1 major vessel": 1,
    "2 major vessels": 2,
    "3 major vessels": 3,
}
thal_options = {
    "Normal blood flow": 3,
    "Fixed defect": 6,
    "Reversible defect": 7,
}


st.markdown(
    """
    <section class="hero">
        <p class="eyebrow">Multi-Objective Evolutionary Optimization</p>
        <h1>CardioGA Optimizer</h1>
        <p>
            A presentation-ready dashboard for tuning heart disease classifiers with a from-scratch
            genetic algorithm. The search balances two goals at the same time: pushing validation
            accuracy upward while keeping training time low enough for efficient deployment and fast iteration.
        </p>
        <div class="badge-row">
            <span class="badge">Dataset: UCI Heart Disease (Cleveland)</span>
            <span class="badge">Objectives: Accuracy and Training Time</span>
            <span class="badge">Search Strategy: Pareto Front + Tournament Selection</span>
            <span class="badge">UI: Streamlit + Plotly</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Experiment Controls")
    st.caption("Tune the search pressure, then launch a full GA run with live progress updates.")
    selected_model = st.selectbox(
        "Model family",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda option: MODEL_LABELS[option],
    )
    population_size = st.slider("Population size", min_value=12, max_value=60, value=28, step=2)
    generations = st.slider("Generations", min_value=6, max_value=35, value=18, step=1)
    mutation_rate = st.slider("Mutation rate", min_value=0.05, max_value=0.45, value=0.18, step=0.01)
    run_button = st.button("Run Genetic Optimization", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### Demo Notes")
    st.write(
        "The GA evaluates model hyperparameters on an internal validation split. "
        "A separate held-out test split is reserved for the final default-vs-optimized comparison."
    )
    st.write(
        "Dominance means one solution is at least as accurate and no slower than another, "
        "with at least one strict improvement. Non-dominated solutions form the Pareto front."
    )
    st.write(
        "The Prediction tab turns the optimized model into a patient-level demo by accepting clinical inputs "
        "and returning a heart disease risk estimate."
    )


if run_button:
    st.session_state["latest_prediction"] = None
    progress_bar = st.progress(0, text="Preparing optimization run...")
    status_box = st.empty()

    def progress_callback(payload: dict) -> None:
        progress_fraction = int((payload["generation"] / payload["total_generations"]) * 100)
        progress_bar.progress(
            progress_fraction,
            text=(
                f"Generation {payload['generation']} / {payload['total_generations']}  |  "
                f"Best accuracy: {payload['best_accuracy'] * 100:.2f}%  |  "
                f"Best time: {payload['best_training_time'] * 1000:.2f} ms"
            ),
        )
        status_box.markdown(
            f"""
            <div class="info-card">
                <h3>Optimization in progress</h3>
                <p>
                    Generation <b>{payload['generation']}</b> is complete. The current leading candidate
                    has <b>{payload['best_accuracy'] * 100:.2f}%</b> validation accuracy with a
                    <b>{payload['best_training_time'] * 1000:.2f} ms</b> training time and a
                    composite fitness of <b>{payload['best_fitness']:.4f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner("Running the evolutionary search and updating the Pareto archive..."):
        configuration = ExperimentConfig(
            model_name=selected_model,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        st.session_state["experiment_result"] = run_experiment(configuration, progress_callback=progress_callback)

    progress_bar.progress(100, text="Optimization complete.")
    status_box.markdown(
        """
        <div class="info-card">
            <h3>Run complete</h3>
            <p>
                The dashboard below now shows the selected GA solution, the explored Pareto front,
                and a direct comparison against the model's untuned baseline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


result = st.session_state["experiment_result"]
multioutput_result = st.session_state["multioutput_result"]

tab_overview, tab_results, tab_graphs, tab_prediction, tab_multioutput = st.tabs(
    ["Overview", "Results", "Graphs", "Prediction", "Severity GA"]
)

with tab_overview:
    overview_columns = st.columns(3)
    with overview_columns[0]:
        render_info_card(
            "Problem Statement",
            "Predicting heart disease matters, but a model that is only accurate can still be impractical if it is too slow to train or iterate on. This project treats model quality and training efficiency as joint design goals instead of forcing a single-metric view.",
        )
    with overview_columns[1]:
        render_info_card(
            "Multi-Objective Goal",
            "The optimizer pushes accuracy upward and training time downward at the same time. Because those objectives can conflict, we keep a Pareto front of non-dominated solutions instead of assuming there is only one universally best answer.",
        )
    with overview_columns[2]:
        render_info_card(
            "How GA Solves It",
            "Each chromosome represents a hyperparameter configuration. The GA samples a population, evaluates model performance, selects strong candidates through tournaments, mixes parents with crossover, injects diversity through mutation, and keeps elite solutions across generations.",
        )

    st.markdown('<div class="section-title">Data Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The loader reads the official processed Cleveland file, converts missing markers to nulls, imputes missing values inside a preprocessing pipeline, one-hot encodes categorical variables, scales numeric features for logistic regression, and creates stratified train, validation, and test splits for a reliable demonstration setup.</div>',
        unsafe_allow_html=True,
    )

    if result:
        profile = result["dataset_profile"]
        cards = st.columns(4)
        with cards[0]:
            render_metric_card("Records", f"{profile['rows']}", "Patients in the Cleveland subset", "tone-fitness")
        with cards[1]:
            render_metric_card("Features", f"{profile['feature_count']}", "Clinical attributes used for modeling", "tone-fitness")
        with cards[2]:
            render_metric_card("Positive Rate", f"{profile['positive_rate'] * 100:.1f}%", "Patients with diagnosed disease", "tone-accuracy")
        with cards[3]:
            render_metric_card("Missing Values", f"{sum(profile['missing_values'].values())}", "Imputed during preprocessing", "tone-time")

        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(result["dataset_preview"], use_container_width=True, hide_index=True)
    else:
        cards = st.columns(4)
        with cards[0]:
            render_metric_card("Records", f"{initial_profile['rows']}", "Patients in the Cleveland subset", "tone-fitness")
        with cards[1]:
            render_metric_card("Features", f"{initial_profile['feature_count']}", "Clinical attributes used for modeling", "tone-fitness")
        with cards[2]:
            render_metric_card("Positive Rate", f"{initial_profile['positive_rate'] * 100:.1f}%", "Patients with diagnosed disease", "tone-accuracy")
        with cards[3]:
            render_metric_card("Missing Values", f"{sum(initial_profile['missing_values'].values())}", "Imputed during preprocessing", "tone-time")

        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(dataset_frame.drop(columns=["target_raw"]).head(8), use_container_width=True, hide_index=True)
        st.info("Run the optimizer from the sidebar to populate the comparisons, Pareto front, and generation charts.")


with tab_results:
    if not result:
        st.warning("No experiment has been executed yet. Launch a run from the sidebar to populate this section.")
    else:
        default_metrics = result["default_metrics"]
        optimized_metrics = result["optimized_metrics"]
        improvement = result["tradeoff_improvement"]

        metric_columns = st.columns(3)
        with metric_columns[0]:
            render_metric_card(
                "Optimized Test Accuracy",
                format_percent(optimized_metrics["test_accuracy"]),
                f"Change vs default: {improvement['test_accuracy_change_pct']:+.2f}%",
                "tone-accuracy",
            )
        with metric_columns[1]:
            render_metric_card(
                "Optimized Test Training Time",
                format_seconds(optimized_metrics["test_training_time"]),
                f"Efficiency change vs default: {improvement['test_time_change_pct']:+.2f}%",
                "tone-time",
            )
        with metric_columns[2]:
            render_metric_card(
                "Best Search Fitness",
                f"{optimized_metrics['fitness']:.4f}",
                "Composite score used for dashboard selection on the Pareto front",
                "tone-fitness",
            )

        detail_columns = st.columns([1.25, 0.95])
        with detail_columns[0]:
            st.markdown('<div class="section-title">Trade-Off Comparison</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-copy">{result["summary_text"]}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(result["comparison_table"], use_container_width=True, hide_index=True)

        with detail_columns[1]:
            st.markdown('<div class="section-title">Selected Hyperparameters</div>', unsafe_allow_html=True)
            st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
            st.json(result["best_hyperparameters"], expanded=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Top Pareto Candidates</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">These are non-dominated configurations discovered during the evolutionary search. They represent the efficient frontier of accuracy and training time rather than just the single dashboard-selected winner.</div>',
            unsafe_allow_html=True,
        )
        top_pareto = result["pareto_front"][
            [
                "generation",
                "validation_accuracy",
                "training_time",
                "composite_fitness",
            ]
            + list(result["best_hyperparameters"].keys())
        ].copy()
        top_pareto["validation_accuracy"] = top_pareto["validation_accuracy"].map(lambda value: f"{value * 100:.2f}%")
        top_pareto["training_time"] = top_pareto["training_time"].map(format_seconds)
        top_pareto["composite_fitness"] = top_pareto["composite_fitness"].map(lambda value: f"{value:.4f}")
        st.dataframe(top_pareto.head(10), use_container_width=True, hide_index=True)


with tab_graphs:
    if not result:
        st.warning("Run an experiment to generate the optimization charts.")
    else:
        history = result["history"]

        graph_top = st.columns(2)
        with graph_top[0]:
            accuracy_figure = create_generation_figure(
                history_frame=history,
                metric_column="best_accuracy",
                title="Accuracy Across Generations",
                y_axis_title="Validation Accuracy (%)",
                color="rgba(83, 211, 166, 1)",
                percentage_axis=True,
            )
            st.plotly_chart(accuracy_figure, use_container_width=True)

        with graph_top[1]:
            time_figure = create_generation_figure(
                history_frame=history.assign(best_training_time_ms=history["best_training_time"] * 1000),
                metric_column="best_training_time_ms",
                title="Training Time Across Generations",
                y_axis_title="Training Time (ms)",
                color="rgba(242, 179, 91, 1)",
                percentage_axis=False,
            )
            st.plotly_chart(time_figure, use_container_width=True)

        graph_bottom = st.columns(2)
        with graph_bottom[0]:
            fitness_figure = create_generation_figure(
                history_frame=history,
                metric_column="best_fitness",
                title="Composite Fitness Across Generations",
                y_axis_title="Fitness Score",
                color="rgba(139, 211, 230, 1)",
                percentage_axis=False,
            )
            st.plotly_chart(fitness_figure, use_container_width=True)

        with graph_bottom[1]:
            pareto_figure = create_pareto_figure(
                all_candidates=result["all_candidates"],
                pareto_candidates=result["pareto_front"],
                default_metrics=result["default_metrics"],
                optimized_metrics=result["optimized_metrics"],
            )
            st.plotly_chart(pareto_figure, use_container_width=True)

        st.markdown('<div class="section-title">Why the Pareto Plot Matters</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Moving upward means higher validation accuracy. Moving left means lower training time. Any point on the Pareto front is efficient: improving one objective further would require sacrificing the other. The dashboard marks the untuned baseline and the selected GA solution directly on this frontier.</div>',
            unsafe_allow_html=True,
        )


with tab_prediction:
    use_optimized_model = bool(result) and result["model_name"] == selected_model
    if use_optimized_model:
        prediction_pipeline = result["prediction_pipeline"]
        active_model_label = result["model_label"]
        active_hyperparameters = result["best_hyperparameters"]
        source_caption = "Using the GA-optimized model retrained on the full dataset for patient-level inference."
    else:
        prediction_pipeline, active_hyperparameters = load_prediction_pipeline(selected_model)
        active_model_label = MODEL_LABELS[selected_model]
        source_caption = "Using the model's default hyperparameters. Run the optimizer first to upgrade this tab to the Pareto-selected configuration."

    prediction_info_columns = st.columns(3)
    with prediction_info_columns[0]:
        render_info_card(
            "Prediction Goal",
            "This tab predicts the likelihood of heart disease presence from patient features. It does not identify a specific named heart condition because the UCI target supports presence/severity rather than disease names.",
        )
    with prediction_info_columns[1]:
        render_info_card(
            "Active Inference Model",
            f"{active_model_label}. {source_caption}",
        )
    with prediction_info_columns[2]:
        render_info_card(
            "How to Explain It",
            "The GA still matters here because the patient prediction uses the model configuration selected by the multi-objective search. The optimization tab finds the best trade-off, and this tab turns that model into an interactive demo.",
        )

    if result and result["model_name"] != selected_model:
        st.info(
            f"The last optimization run was for {result['model_label']}, but the sidebar is currently set to "
            f"{MODEL_LABELS[selected_model]}. The prediction form is using the current sidebar model with default hyperparameters."
        )

    st.markdown('<div class="section-title">Patient Detail Entry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Enter the patient profile below. The app will preprocess the inputs exactly like the training pipeline and return a probability of heart disease presence. This is a model demonstration, not a medical diagnosis.</div>',
        unsafe_allow_html=True,
    )

    with st.form("patient_prediction_form"):
        row_one = st.columns(3)
        with row_one[0]:
            age = st.number_input("Age", min_value=20, max_value=100, value=patient_defaults["age"], step=1)
            sex_label = st.selectbox(
                "Sex",
                options=list(sex_options.keys()),
                index=get_option_index(sex_options, patient_defaults["sex"]),
            )
            cp_label = st.selectbox(
                "Chest Pain Type",
                options=list(cp_options.keys()),
                index=get_option_index(cp_options, patient_defaults["cp"]),
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=240,
                value=patient_defaults["trestbps"],
                step=1,
            )

        with row_one[1]:
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100,
                max_value=700,
                value=patient_defaults["chol"],
                step=1,
            )
            fbs_label = st.selectbox(
                "Fasting Blood Sugar",
                options=list(fbs_options.keys()),
                index=get_option_index(fbs_options, patient_defaults["fbs"]),
            )
            restecg_label = st.selectbox(
                "Resting ECG",
                options=list(restecg_options.keys()),
                index=get_option_index(restecg_options, patient_defaults["restecg"]),
            )
            thalach = st.number_input(
                "Maximum Heart Rate",
                min_value=60,
                max_value=230,
                value=patient_defaults["thalach"],
                step=1,
            )

        with row_one[2]:
            exang_label = st.selectbox(
                "Exercise-Induced Angina",
                options=list(exang_options.keys()),
                index=get_option_index(exang_options, patient_defaults["exang"]),
            )
            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=7.0,
                value=patient_defaults["oldpeak"],
                step=0.1,
                format="%.1f",
            )
            slope_label = st.selectbox(
                "ST Segment Slope",
                options=list(slope_options.keys()),
                index=get_option_index(slope_options, patient_defaults["slope"]),
            )
            ca_label = st.selectbox(
                "Number of Major Vessels",
                options=list(ca_options.keys()),
                index=get_option_index(ca_options, patient_defaults["ca"]),
            )
            thal_label = st.selectbox(
                "Thalassemia Result",
                options=list(thal_options.keys()),
                index=get_option_index(thal_options, patient_defaults["thal"]),
            )

        predict_button = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True, type="primary")

    if predict_button:
        patient_values = {
            "age": int(age),
            "sex": sex_options[sex_label],
            "cp": cp_options[cp_label],
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": fbs_options[fbs_label],
            "restecg": restecg_options[restecg_label],
            "thalach": int(thalach),
            "exang": exang_options[exang_label],
            "oldpeak": float(oldpeak),
            "slope": slope_options[slope_label],
            "ca": ca_options[ca_label],
            "thal": thal_options[thal_label],
        }
        patient_frame = build_patient_dataframe(patient_values)
        predicted_class = int(prediction_pipeline.predict(patient_frame)[0])
        predicted_probability = float(prediction_pipeline.predict_proba(patient_frame)[0][1])
        risk_band, card_tone, narrative = infer_risk_band(predicted_probability)
        st.session_state["latest_prediction"] = {
            "patient_values": patient_values,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_probability,
            "risk_band": risk_band,
            "card_tone": card_tone,
            "narrative": narrative,
            "model_name": selected_model,
            "uses_optimized_model": use_optimized_model,
            "active_model_label": active_model_label,
            "source_caption": source_caption,
            "active_hyperparameters": active_hyperparameters,
        }

    prediction_result = st.session_state["latest_prediction"]
    if prediction_result and (
        prediction_result["model_name"] != selected_model
        or prediction_result["uses_optimized_model"] != use_optimized_model
    ):
        st.session_state["latest_prediction"] = None
        prediction_result = None

    if prediction_result:
        outcome_text = "Disease Likely" if prediction_result["predicted_class"] == 1 else "Lower Risk"
        metrics = st.columns(3)
        with metrics[0]:
            render_metric_card(
                "Predicted Outcome",
                outcome_text,
                prediction_result["narrative"],
                prediction_result["card_tone"],
            )
        with metrics[1]:
            render_metric_card(
                "Predicted Probability",
                format_percent(prediction_result["predicted_probability"]),
                "Estimated probability of heart disease presence for the entered profile",
                "tone-accuracy" if prediction_result["predicted_probability"] < 0.5 else "tone-time",
            )
        with metrics[2]:
            render_metric_card(
                "Inference Model",
                prediction_result["active_model_label"],
                prediction_result["risk_band"],
                "tone-fitness",
            )

        st.progress(
            int(round(prediction_result["predicted_probability"] * 100)),
            text=f"Risk score: {prediction_result['predicted_probability'] * 100:.2f}% probability of heart disease presence",
        )

        summary_columns = st.columns([1.1, 0.9])
        with summary_columns[0]:
            st.markdown('<div class="section-title">Prediction Interpretation</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-copy">{prediction_result["source_caption"]} '
                f'The current profile falls into the <b>{prediction_result["risk_band"]}</b> band. '
                'Use this as a model explanation for demonstration purposes, not as a clinical diagnosis.</div>',
                unsafe_allow_html=True,
            )
            patient_summary = pd.DataFrame(
                [{"Feature": key, "Value": value} for key, value in prediction_result["patient_values"].items()]
            )
            st.dataframe(patient_summary, use_container_width=True, hide_index=True)

        with summary_columns[1]:
            st.markdown('<div class="section-title">Model Configuration Used</div>', unsafe_allow_html=True)
            st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
            st.json(prediction_result["active_hyperparameters"], expanded=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Submit the patient form to generate a live heart disease risk prediction.")


with tab_multioutput:
    intro_cols = st.columns(3)
    with intro_cols[0]:
        render_info_card(
            "Your Original ML Logic",
            "This tab uses your MultiOutputClassifier idea: one Random Forest predicts the original severity class, and the second output predicts the derived risk level.",
        )
    with intro_cols[1]:
        render_info_card(
            "GA Chromosome",
            "Here, each chromosome is a binary feature mask. A gene value of 1 keeps that feature, and 0 removes it from the Random Forest training run.",
        )
    with intro_cols[2]:
        render_info_card(
            "Multi-Objective Score",
            "The fitness score rewards average severity-risk accuracy and penalizes training time, so the search looks for accurate feature subsets that train efficiently.",
        )

    st.markdown('<div class="section-title">Run Multi-Output Feature Selection GA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">This workflow mirrors your earlier GA code but makes it dashboard-ready: baseline Random Forest, GA-selected feature subset, generation history, and final patient prediction for severity plus risk level.</div>',
        unsafe_allow_html=True,
    )

    control_cols = st.columns(3)
    with control_cols[0]:
        multi_population_size = st.slider(
            "Severity GA population",
            min_value=6,
            max_value=30,
            value=12,
            step=2,
            key="multi_population_size",
        )
    with control_cols[1]:
        multi_generations = st.slider(
            "Severity GA generations",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            key="multi_generations",
        )
    with control_cols[2]:
        multi_mutation_rate = st.slider(
            "Severity GA mutation rate",
            min_value=0.05,
            max_value=0.45,
            value=0.20,
            step=0.01,
            key="multi_mutation_rate",
        )

    run_multioutput_button = st.button(
        "Run Multi-Output GA",
        use_container_width=True,
        type="primary",
        key="run_multioutput_ga_button",
    )

    if run_multioutput_button:
        st.session_state["latest_multioutput_prediction"] = None
        progress_bar = st.progress(0, text="Preparing multi-output GA...")
        status_box = st.empty()

        def multioutput_progress(payload: dict) -> None:
            progress_percent = int((payload["generation"] / payload["total_generations"]) * 100)
            progress_bar.progress(
                progress_percent,
                text=(
                    f"Generation {payload['generation']} / {payload['total_generations']}  |  "
                    f"Severity: {payload['severity_accuracy'] * 100:.2f}%  |  "
                    f"Risk: {payload['risk_accuracy'] * 100:.2f}%  |  "
                    f"Features: {payload['selected_features']}"
                ),
            )
            status_box.markdown(
                f"""
                <div class="info-card">
                    <h3>Feature-selection GA running</h3>
                    <p>
                        Current best fitness is <b>{payload['best_fitness']:.4f}</b>.
                        The selected chromosome keeps <b>{payload['selected_features']}</b> encoded features.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.spinner("Running feature-selection GA for severity and risk-level prediction..."):
            multioutput_config = MultiOutputGAConfig(
                population_size=multi_population_size,
                generations=multi_generations,
                mutation_rate=multi_mutation_rate,
            )
            st.session_state["multioutput_result"] = run_multioutput_feature_ga(
                multioutput_config,
                progress_callback=multioutput_progress,
            )
            multioutput_result = st.session_state["multioutput_result"]

        progress_bar.progress(100, text="Multi-output GA complete.")
        status_box.success("Multi-output severity/risk model is ready for patient prediction.")

    if not multioutput_result:
        st.info("Run the Multi-Output GA above to unlock severity, risk-level metrics, selected features, and patient prediction.")
    else:
        baseline = multioutput_result["baseline_metrics"]
        optimized = multioutput_result["optimized_metrics"]

        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card(
                "Severity Accuracy",
                format_percent(optimized["severity_accuracy"]),
                f"Baseline: {format_percent(baseline['severity_accuracy'])}",
                "tone-accuracy",
            )
        with metric_cols[1]:
            render_metric_card(
                "Risk Accuracy",
                format_percent(optimized["risk_accuracy"]),
                f"Baseline: {format_percent(baseline['risk_accuracy'])}",
                "tone-accuracy",
            )
        with metric_cols[2]:
            render_metric_card(
                "Training Time",
                format_seconds(optimized["training_time"]),
                f"Baseline: {format_seconds(baseline['training_time'])}",
                "tone-time",
            )
        with metric_cols[3]:
            render_metric_card(
                "Selected Features",
                f"{multioutput_result['selected_feature_count']}",
                f"Out of {multioutput_result['total_feature_count']} encoded features",
                "tone-fitness",
            )

        comparison = pd.DataFrame(
            [
                {
                    "Metric": "Severity Accuracy",
                    "Baseline RF": format_percent(baseline["severity_accuracy"]),
                    "GA Feature RF": format_percent(optimized["severity_accuracy"]),
                },
                {
                    "Metric": "Risk Level Accuracy",
                    "Baseline RF": format_percent(baseline["risk_accuracy"]),
                    "GA Feature RF": format_percent(optimized["risk_accuracy"]),
                },
                {
                    "Metric": "Exact Match Accuracy",
                    "Baseline RF": format_percent(baseline["exact_match_accuracy"]),
                    "GA Feature RF": format_percent(optimized["exact_match_accuracy"]),
                },
                {
                    "Metric": "Training Time",
                    "Baseline RF": format_seconds(baseline["training_time"]),
                    "GA Feature RF": format_seconds(optimized["training_time"]),
                },
            ]
        )
        st.markdown('<div class="section-title">Before vs After GA</div>', unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        charts = st.columns(2)
        with charts[0]:
            severity_history = multioutput_result["history"].rename(
                columns={"best_severity_accuracy": "severity_accuracy"}
            )
            severity_figure = create_generation_figure(
                severity_history,
                "severity_accuracy",
                "Severity Accuracy Across Generations",
                "Severity Accuracy (%)",
                "rgba(83, 211, 166, 1)",
                percentage_axis=True,
            )
            st.plotly_chart(severity_figure, use_container_width=True)
        with charts[1]:
            risk_history = multioutput_result["history"].rename(columns={"best_risk_accuracy": "risk_accuracy"})
            risk_figure = create_generation_figure(
                risk_history,
                "risk_accuracy",
                "Risk-Level Accuracy Across Generations",
                "Risk-Level Accuracy (%)",
                "rgba(139, 211, 230, 1)",
                percentage_axis=True,
            )
            st.plotly_chart(risk_figure, use_container_width=True)

        lower_charts = st.columns(2)
        with lower_charts[0]:
            fitness_figure = create_generation_figure(
                multioutput_result["history"],
                "best_fitness",
                "Multi-Output Fitness Across Generations",
                "Fitness Score",
                "rgba(242, 179, 91, 1)",
                percentage_axis=False,
            )
            st.plotly_chart(fitness_figure, use_container_width=True)
        with lower_charts[1]:
            selected_features_history = multioutput_result["history"].rename(
                columns={"selected_features": "selected_feature_count"}
            )
            feature_figure = create_generation_figure(
                selected_features_history,
                "selected_feature_count",
                "Feature Count Across Generations",
                "Selected Encoded Features",
                "rgba(255, 122, 89, 1)",
                percentage_axis=False,
            )
            st.plotly_chart(feature_figure, use_container_width=True)

        st.markdown('<div class="section-title">Selected Encoded Features</div>', unsafe_allow_html=True)
        feature_frame = pd.DataFrame({"Selected Features": multioutput_result["selected_feature_names"]})
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Predict Severity and Risk Level</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">This form uses the GA-selected feature subset and final multi-output Random Forest. It predicts the UCI severity class and the derived risk level, not a named clinical disease diagnosis.</div>',
            unsafe_allow_html=True,
        )

        with st.form("multioutput_patient_prediction_form"):
            first_row = st.columns(3)
            with first_row[0]:
                mo_age = st.number_input("Age", min_value=20, max_value=100, value=patient_defaults["age"], step=1, key="mo_age")
                mo_sex_label = st.selectbox(
                    "Sex",
                    options=list(sex_options.keys()),
                    index=get_option_index(sex_options, patient_defaults["sex"]),
                    key="mo_sex",
                )
                mo_cp_label = st.selectbox(
                    "Chest Pain Type",
                    options=list(cp_options.keys()),
                    index=get_option_index(cp_options, patient_defaults["cp"]),
                    key="mo_cp",
                )
                mo_trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=240,
                    value=patient_defaults["trestbps"],
                    step=1,
                    key="mo_trestbps",
                )
            with first_row[1]:
                mo_chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=700,
                    value=patient_defaults["chol"],
                    step=1,
                    key="mo_chol",
                )
                mo_fbs_label = st.selectbox(
                    "Fasting Blood Sugar",
                    options=list(fbs_options.keys()),
                    index=get_option_index(fbs_options, patient_defaults["fbs"]),
                    key="mo_fbs",
                )
                mo_restecg_label = st.selectbox(
                    "Resting ECG",
                    options=list(restecg_options.keys()),
                    index=get_option_index(restecg_options, patient_defaults["restecg"]),
                    key="mo_restecg",
                )
                mo_thalach = st.number_input(
                    "Maximum Heart Rate",
                    min_value=60,
                    max_value=230,
                    value=patient_defaults["thalach"],
                    step=1,
                    key="mo_thalach",
                )
            with first_row[2]:
                mo_exang_label = st.selectbox(
                    "Exercise-Induced Angina",
                    options=list(exang_options.keys()),
                    index=get_option_index(exang_options, patient_defaults["exang"]),
                    key="mo_exang",
                )
                mo_oldpeak = st.number_input(
                    "ST Depression (Oldpeak)",
                    min_value=0.0,
                    max_value=7.0,
                    value=patient_defaults["oldpeak"],
                    step=0.1,
                    format="%.1f",
                    key="mo_oldpeak",
                )
                mo_slope_label = st.selectbox(
                    "ST Segment Slope",
                    options=list(slope_options.keys()),
                    index=get_option_index(slope_options, patient_defaults["slope"]),
                    key="mo_slope",
                )
                mo_ca_label = st.selectbox(
                    "Number of Major Vessels",
                    options=list(ca_options.keys()),
                    index=get_option_index(ca_options, patient_defaults["ca"]),
                    key="mo_ca",
                )
                mo_thal_label = st.selectbox(
                    "Thalassemia Result",
                    options=list(thal_options.keys()),
                    index=get_option_index(thal_options, patient_defaults["thal"]),
                    key="mo_thal",
                )

            multi_predict_button = st.form_submit_button(
                "Predict Severity and Risk Level",
                use_container_width=True,
                type="primary",
            )

        if multi_predict_button:
            multi_patient_values = {
                "age": int(mo_age),
                "sex": sex_options[mo_sex_label],
                "cp": cp_options[mo_cp_label],
                "trestbps": int(mo_trestbps),
                "chol": int(mo_chol),
                "fbs": fbs_options[mo_fbs_label],
                "restecg": restecg_options[mo_restecg_label],
                "thalach": int(mo_thalach),
                "exang": exang_options[mo_exang_label],
                "oldpeak": float(mo_oldpeak),
                "slope": slope_options[mo_slope_label],
                "ca": ca_options[mo_ca_label],
                "thal": thal_options[mo_thal_label],
            }
            st.session_state["latest_multioutput_prediction"] = predict_multioutput_patient(
                multioutput_result["prediction_bundle"],
                multi_patient_values,
            )

        multi_prediction = st.session_state["latest_multioutput_prediction"]
        if multi_prediction:
            prediction_cols = st.columns(3)
            with prediction_cols[0]:
                render_metric_card(
                    "Predicted Severity",
                    f"Class {multi_prediction['severity_class']}",
                    multi_prediction["severity_label"],
                    "tone-time" if multi_prediction["severity_class"] >= 3 else "tone-accuracy",
                )
            with prediction_cols[1]:
                render_metric_card(
                    "Predicted Risk Level",
                    multi_prediction["risk_label"],
                    f"Risk confidence: {format_percent(multi_prediction['risk_probability'])}",
                    "tone-time" if multi_prediction["risk_level"] == 2 else "tone-fitness",
                )
            with prediction_cols[2]:
                render_metric_card(
                    "Severity Confidence",
                    format_percent(multi_prediction["severity_probability"]),
                    "Probability assigned to the predicted UCI severity class",
                    "tone-fitness",
                )
        else:
            st.info("Submit the severity form to generate multi-output predictions from the GA-selected feature model.")


if result:
    with st.expander("Show raw result payload for demo narration"):
        st.code(
            json.dumps(
                {
                    "model": result["model_label"],
                    "completed_generations": result["completed_generations"],
                    "stopped_early": result["stopped_early"],
                    "default_test_accuracy": round(result["default_metrics"]["test_accuracy"], 4),
                    "optimized_test_accuracy": round(result["optimized_metrics"]["test_accuracy"], 4),
                    "default_test_training_time": round(result["default_metrics"]["test_training_time"], 6),
                    "optimized_test_training_time": round(result["optimized_metrics"]["test_training_time"], 6),
                    "best_hyperparameters": result["best_hyperparameters"],
                },
                indent=2,
            ),
            language="json",
        )
