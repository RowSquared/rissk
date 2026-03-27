"""Project pipelines."""

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from kedro.pipeline import Pipeline, node, pipeline

from rissk_kedro.pipelines.feature_creation.nodes import make_qnr_filter


def _load_questionnaire_names() -> list[str]:
    """Read questionnaire names from conf/base/globals.yml at registry build time.

    pipeline_registry.py is imported before Kedro's ConfigLoader is available, so
    globals.yml is read directly via yaml.safe_load.  The path is resolved relative
    to this file: src/rissk_kedro/ -> (parents[2]) -> rissk_kedro/ project root.
    """
    globals_path = Path(__file__).parents[2] / "conf" / "base" / "globals.yml"
    with globals_path.open() as fh:
        globals_data = yaml.safe_load(fh)
    questionnaires = globals_data.get("survey", {}).get("questionnaires", [])
    return [q["name"] for q in questionnaires]


def _make_merge_node(
    output_name: str,
    input_names: list[str],
    node_name: str,
) -> node:
    """Build a node that pd.concat-s N MemoryDataset DataFrames into one output."""
    n = len(input_names)

    def merge_fn(*dfs):
        non_empty = [df for df in dfs if df is not None and not df.empty]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True)

    # Give the function a unique __name__ so Kedro uses it in the node label.
    merge_fn.__name__ = node_name

    return node(
        func=merge_fn,
        inputs=input_names,
        outputs=output_name,
        name=node_name,
    )


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Builds one filter + namespaced-scoring pipeline instance per questionnaire,
    then adds a merge pipeline that concatenates per-questionnaire scored outputs
    back into the same three catalog datasets (item_scores, unit_risk_scores,
    responsible_scores) that exist today.  Catalog is unchanged.
    """
    # Import sub-pipelines here to avoid circular imports at module level.
    from rissk_kedro.pipelines.data_ingestion import create_pipeline as ingestion_pipeline
    from rissk_kedro.pipelines.feature_engineering import create_pipeline as feature_engineering_pipeline
    from rissk_kedro.pipelines.feature_creation import create_pipeline as feature_creation_pipeline
    from rissk_kedro.pipelines.rissk_scoring import create_pipeline as scoring_pipeline

    qnr_names = _load_questionnaire_names()

    # ------------------------------------------------------------------ #
    # Per-questionnaire filter + scoring pipelines                        #
    # ------------------------------------------------------------------ #
    per_qnr_pipelines: dict[str, Pipeline] = {}

    item_score_datasets: list[str] = []
    unit_score_datasets: list[str] = []
    resp_score_datasets: list[str] = []

    for qnr_name in qnr_names:
        # Sanitise the questionnaire name so it is a valid Python identifier /
        # Kedro namespace component (spaces -> underscores, etc.).
        ns = qnr_name.replace(" ", "_").replace("-", "_")

        # -- Filter node --------------------------------------------------
        filter_node = node(
            func=make_qnr_filter(qnr_name),
            inputs=["item_features", "unit_features", "removed_answers"],
            outputs=[
                f"item_features__{ns}",
                f"unit_features__{ns}",
                f"removed_answers__{ns}",
            ],
            name=f"filter_features_{ns}_node",
        )

        # -- Namespaced scoring pipeline ----------------------------------
        # Explicit input/output mappings override namespacing for those keys so
        # the filter outputs wire directly and the final scored dfs get unique names.
        # parameters must be passed via the dedicated `parameters` arg — Kedro
        # raises PipelineError if they appear in `inputs`.
        namespaced_scoring = pipeline(
            scoring_pipeline(),
            namespace=ns,
            inputs={
                "item_features": f"item_features__{ns}",
                "unit_features": f"unit_features__{ns}",
                "removed_answers": f"removed_answers__{ns}",
            },
            parameters={"parameters": "parameters"},
            outputs={
                "item_scores": f"item_scores__{ns}",
                "unit_risk_scores": f"unit_risk_scores__{ns}",
                "responsible_scores": f"responsible_scores__{ns}",
            },
        )

        item_score_datasets.append(f"item_scores__{ns}")
        unit_score_datasets.append(f"unit_risk_scores__{ns}")
        resp_score_datasets.append(f"responsible_scores__{ns}")

        qnr_pipeline = Pipeline([filter_node]) + namespaced_scoring
        per_qnr_pipelines[f"scoring_{ns}"] = qnr_pipeline

    # ------------------------------------------------------------------ #
    # Merge pipeline — concat all per-qnr outputs into catalog datasets  #
    # ------------------------------------------------------------------ #
    merge_pipeline = Pipeline([
        _make_merge_node("item_scores", item_score_datasets, "merge_item_scores_node"),
        _make_merge_node("unit_risk_scores", unit_score_datasets, "merge_unit_scores_node"),
        _make_merge_node("responsible_scores", resp_score_datasets, "merge_responsible_scores_node"),
    ])

    # ------------------------------------------------------------------ #
    # Shared upstream pipelines                                           #
    # ------------------------------------------------------------------ #
    ingestion = ingestion_pipeline()
    feat_eng = feature_engineering_pipeline()
    feat_creation = feature_creation_pipeline()

    all_scoring = sum(per_qnr_pipelines.values(), Pipeline([])) + merge_pipeline

    pipelines: dict[str, Pipeline] = {}

    # Named pipelines for selective runs
    pipelines["data_ingestion"] = ingestion
    pipelines["feature_engineering"] = feat_eng
    pipelines["feature_creation"] = feat_creation
    pipelines["scoring"] = all_scoring   # filter + score + merge; skips ingestion/feature creation

    # Individual per-questionnaire scoring (without merge) — useful for debugging
    for name, p in per_qnr_pipelines.items():
        pipelines[name] = p

    # Full run
    pipelines["__default__"] = ingestion + feat_eng + feat_creation + all_scoring

    return pipelines
