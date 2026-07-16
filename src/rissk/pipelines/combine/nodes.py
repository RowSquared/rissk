"""Nodes for the combine pipeline."""
import logging
from typing import Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def combine_microdata_node(partitions: Dict[str, Callable[[], pd.DataFrame]]) -> pd.DataFrame:
    """Union the per-questionnaire microdata into one survey-level table.

    ``partitions`` is a PartitionedDataset mapping of partition-key -> loader over
    ``30_PROCESSED``. The top-level union file (partition key ``''``) is skipped so the
    output can be rewritten in place idempotently; each ``'<qnr>/'`` partition is a
    per-questionnaire ``microdata.parquet``.
    """
    frames = []
    for key, load in sorted(partitions.items()):
        if not key.strip("/"):
            continue  # the survey-level union file itself — never fold it back in
        frames.append(load())
        logger.info("combine_microdata: adding partition %r", key.strip("/"))

    if not frames:
        logger.warning(
            "combine_microdata: no per-questionnaire microdata partitions found — "
            "returning empty DataFrame."
        )
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("combine_microdata: unioned %d partitions -> %d rows", len(frames), len(combined))
    return combined
