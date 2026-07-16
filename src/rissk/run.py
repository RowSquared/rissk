"""In-process driver for running the pipeline across a survey's questionnaires.

One survey = one Kedro env (``conf/<env>/``). The questionnaires belonging to that
survey are declared as one small YAML each under ``conf/<env>/questionnaires/``. This
module enumerates them and runs the static 3-stage pipeline once per questionnaire
(injecting the selection via runtime params), then runs the ``combine`` pipeline once.

A legacy env with a single questionnaire in ``globals.yml`` and no ``questionnaires/``
folder is run once, unchanged, with no combine step.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import yaml

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _project_root(project_root: Optional[PathLike] = None) -> Path:
    if project_root is not None:
        return Path(project_root)
    import rissk
    return Path(rissk.__file__).resolve().parents[2]


def load_questionnaire_configs(env: str, project_root: Optional[PathLike] = None) -> list[dict]:
    """Parse ``conf/<env>/questionnaires/*.yml`` (sorted by filename). Empty if absent."""
    qdir = _project_root(project_root) / "conf" / env / "questionnaires"
    if not qdir.is_dir():
        return []
    configs = []
    for path in sorted(qdir.glob("*.yml")):
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        if "name" not in cfg:
            raise ValueError(f"{path}: questionnaire yaml must set 'name'")
        configs.append(cfg)
    return configs


def run_survey(
    env: str,
    project_root: Optional[PathLike] = None,
    pipeline: str = "__default__",
    run_combine: bool = True,
) -> dict[str, str]:
    """Run the pipeline for every questionnaire in ``env``, then combine once.

    Returns an outcome map; failures are isolated per questionnaire so one bad
    questionnaire does not skip the rest.
    """
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    root = _project_root(project_root)
    bootstrap_project(root)
    qnrs = load_questionnaire_configs(env, root)
    results: dict[str, str] = {}

    def _run(label: str, extra_params: Optional[dict], pipe: str) -> None:
        print(f"=== run --env {env} --pipeline {pipe} [{label}] ===", flush=True)
        try:
            with KedroSession.create(project_path=root, env=env, extra_params=extra_params) as session:
                session.run(pipeline_names=[pipe])
            results[label] = "OK"
        except Exception as exc:  # isolate failures across questionnaires
            results[label] = f"FAILED ({type(exc).__name__}: {exc})"
        print(f"--- {label}: {results[label]} ---", flush=True)

    if not qnrs:
        # Legacy single-questionnaire env: questionnaire comes from globals, no combine.
        _run(env, None, pipeline)
        return results

    for q in qnrs:
        name = q["name"]
        _run(name, {"questionnaire": q, "qnr_subdir": f"{name}/"}, pipeline)

    if run_combine:
        _run("combine", None, "combine")

    return results
