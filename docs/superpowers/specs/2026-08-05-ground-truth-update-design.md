# Ground-truth corpus update

## Goal

Align `benchmark/ground_truth/` with the notebooks the repository currently
keeps for corpus analysis: the eight notebooks under `notebooks/examples/` and
the four self-contained notebooks under `notebooks/instructions/Notebook1/`
through `Notebook4/`.

The demo notebook under `notebooks/demos/` is intentionally outside this
benchmark corpus. It demonstrates the agent workflow rather than serving as a
scientific notebook whose provenance should be judged.

## Ground-truth method

Every record will be produced by manual AI review of the notebook's markdown
and code cells. The repository's dataset-detection agent will not be called to
generate or validate labels. The review will identify:

- top-level software imports, including standard-library imports when they
  appear in the notebook's code, following the existing ground-truth style;
- externally sourced data and the terminal variables that carry each source
  into analysis or the notebook's intended provenance task; and
- the source tool (`LiPDGraph`, `PyLiPD`, `PyleoTUPS`, `xarray`, or another
  loader named by the notebook) plus notes for judgment calls.

The notebooks will not be executed. Network access and generated bibliography
files are outside this update.

## File changes

- Retain and review the seven existing example records:
  `C02_b_DA_with_individual_seasonality`, `CMIP6_LMR`, `VICS_dashboard`,
  `data_from_esm_cloudcat`, `paleoPCAlite`,
  `spatial_snapshots_xarray_bonuses`, and `widget_primer`.
- Add `02a-query_lipd_graph.yml`.
- Add records for `Notebook1` through `Notebook4`.
- Remove records whose notebooks are no longer in the kept corpus:
  `Graph.yml`, `LIPD.yml`, `PyleoTUPS.yml`, `dataset_pipeline.yml`, and
  `paleoPCA.yml`.

The resulting directory will contain 12 YAML records, one per kept corpus
notebook. Each record will retain the existing `notebook`, `software`,
`datasets`, and `notes` structure.

## Validation

After editing:

1. Parse every ground-truth YAML file.
2. Confirm each `notebook` path exists and belongs to `examples/` or
   `instructions/`.
3. Confirm there are exactly 12 records and no record points to a deleted
   notebook.
4. Review the generated records against the notebook code and markdown,
   especially local `.lpd` siblings and LiPDGraph/PyleoTUPS terminal objects.

No package code, tests, notebook contents, or documentation outside the
benchmark records will be changed as part of this update.
