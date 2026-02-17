# Data Ingestion Discrepancies: Ploomber (Legacy) vs Kedro (New)

## Overview
This document tracks intentional data discrepancies between the legacy Ploomber pipeline and the new Kedro pipeline. These differences are accepted improve data quality or cleanliness.

## 1. Microdata `value` Column Normalization

### The Discrepancy
- **Legacy (Ploomber):** The `value` column in `microdata.parquet` contains a mix of format styles for integer-like values.
  - Example: `1` (integer-like string) and `1.0` (float-like string) appear inconsistently for the same logical value.
- **New (Kedro):** The pipeline now explicitly normalizes values before conversion to string.
  - Logic: If a float value `x` is equivalent to an integer (`x.is_integer() is True`), it is converted to an integer before stringification.
  - Result: `1.0` becomes `"1"`. `1.5` remains `"1.5"`.
  - Lists: This normalization is also applied to values inside list-strings (e.g., `"[1.0, 2.0]"` becomes `"[1, 2]"`).

### Decision
**Status:** ACCEPTED (Intentional Deviation)

We have chosen to keep the cleaner, normalized integer format in Kedro. 
- **Reasoning:** 
  1. The values are typically categorical codes (IDs, boolean flags like 0/1), where `1` is semantically more accurate than `1.0`.
  2. Mixed formatting in the legacy pipeline appears to be an artifact of how Pandas handles `NaN`s (forcing floats) rather than intentional data design.
  3. Uniform formatting simplifies downstream processing.

### Downstream Implications
Any downstream code (feature engineering, analysis) that performs **exact string matching** against float-strings (e.g., `val == "1.0"`) may fail or return empty results.
- **Action Required:** Ensure downstream filtering uses type-safe comparisons (convert to float/int before comparing) or checks for the normalized string `"1"`.
