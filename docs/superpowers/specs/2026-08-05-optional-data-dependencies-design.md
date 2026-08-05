# Optional Data Dependencies Design

## Goal

Keep the core provenance-agent installation focused on notebook parsing,
software citation, and dataset detection while allowing users of generated
data-retrieval cells to opt into the paleoclimate providers.

## Design

Remove `pylipd` and `pyleotups` from `[project].dependencies` and add both to
`[project.optional-dependencies].data`. The source code and generated-cell
behavior remain unchanged. Users who execute PyLiPD, LiPDGraph, or PyleoTUPS
retrieval cells install `.[data]` in the notebook kernel's environment; users
of software citations, static detection, or the benchmark runner do not need
those packages.

Update README and documentation-draft installation instructions so the full
development-plus-Google setup uses `.[dev,google,data]`, while the core and
provider-only commands remain valid without the data extra. Add a packaging
metadata test proving the two requirements are conditional on the `data`
extra rather than core requirements.

## Constraints

- Do not change runtime source code or notebook behavior.
- Preserve all unrelated working-tree changes.
- Reinstall the editable package after changing `pyproject.toml` before running
  metadata and full-suite verification.
