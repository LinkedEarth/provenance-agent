# Instruction Notebook Dataset Detection Design

**Date:** 2026-08-03

## Goal

Make the deterministic dataset detector analyze the additional notebooks under
`notebooks/testing/Instruction Notebooks/`. These notebooks use empty code cells
for student answers and store reference implementations in fenced Python blocks
inside Markdown cells.

The detector must keep its existing contract: return only recognized dataset
sources whose lineage reaches a scientific analysis operation, as
`[variable, tool]` pairs, without executing the notebook or calling an LLM.

## Root cause

`deterministic_dataset_detection._parse_code_cells()` currently reads only cells
whose `cell_type` is `code`. The added instruction notebooks therefore present
the detector with placeholder headings instead of the solution code that loads
and analyzes data. When the solution code is scanned separately, the existing
data-flow graph also exposes two vocabulary gaps:

- `wavelet_coherence`, `ruptures`, and Pyleoclim `Series`/`GeoSeries`
  boundaries are not recognized analysis sinks.
- Notebook 2 retrieves a PyleoTUPS study with
  `get_data(study_id=...)` without first calling `search_studies()`, so its
  source object remains inactive under the current conservative activation rule.

## Design

### 1. Treat fenced Python Markdown as virtual source units

Extend the deterministic detector's notebook reader to collect fenced Python
blocks marked with `python` or `py` from Markdown cells. Each block is parsed
independently as Python, skipped if it has a syntax error, and processed in the
same order as the surrounding notebook cells. The blocks are virtual source
units only: the notebook file is not changed, and no code is executed.

The graph must remain continuous across blocks. A loader in one Markdown block
must be available to later blocks, just as a loader in one real code cell is
available to later code cells. Generated provenance cells remain excluded.
Non-Python Markdown fences are ignored.

### 2. Expand the analysis vocabulary used by the fixtures

Add the analysis boundaries exercised by the instruction notebooks:

- `wavelet_coherence`
- `ruptures`
- `Series`
- `GeoSeries`

Matching remains case-insensitive and underscore-insensitive through the
existing normalization helper. These additions only make an existing source
lineage observable; they do not create sources without recognized lineage.

### 3. Recognize direct PyleoTUPS study retrieval

When a recognized PyleoTUPS object calls `get_data()` with an explicit study
identifier (`study_id` or `noaa_id`), mark that source as active before
propagating the returned table lineage. A bare `get_data()` call on an object
that has not been activated by a recognized search/load method remains
conservative and continues to produce the existing unresolved-lineage
diagnostic.

### 4. Regression coverage

Add tests that verify:

- fenced Python blocks in Markdown are scanned in order and can share variable
  lineage;
- non-Python fences and generated cells do not become detector input;
- direct PyleoTUPS study retrieval activates only when an explicit identifier is
  supplied;
- each of the four instruction notebooks returns the expected analysis-used
  sources:

  ```text
  Notebook1: [["df1", "LiPDGraph"], ["ds", "PyleoTUPS"], ["D", "PyLiPD"]]
  Notebook2: [["ds", "PyleoTUPS"], ["lipd", "PyLiPD"]]
  Notebook3: [["dsp", "PyleoTUPS"], ["lipd", "PyLiPD"]]
  Notebook4: [["D_ice", "LiPDGraph"], ["dsp", "PyleoTUPS"], ["D", "PyLiPD"]]
  ```

Notebook 2's graph query and Notebook 3's graph query are intentionally absent
because their resulting tables do not reach a recognized analysis operation in
the reference code. Existing tests for unused or unsupported sources must
continue to pass.

## Files in scope

- `src/deterministic_dataset_detection.py`: virtual source-unit extraction,
  analysis registry additions, and direct PyleoTUPS activation.
- `tests/test_deterministic_dataset_detection.py`: focused regression tests and
  instruction-notebook expectations.

No notebook contents, deprecated LLM helpers, workflow APIs, or citation
retrieval logic are changed.

## Verification

Run the focused detector tests first, then the full test suite in the repository's
`lang` environment. The final verification must also invoke the detector on all
four instruction notebooks and compare the exact pair lists above.
