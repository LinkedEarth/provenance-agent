# Software citations as a DataFrame design

Date: 2026-07-14
Issue: #26 (software citations DataFrame)
Related: #27 (follow-up - data_workflow.build_retrieval_cell currently
discards the metadata DataFrame that get_bibtex()/get_publications() already
return)

## Goal

Make the software workflow's citation-collection step return a
`pandas.DataFrame` instead of a `pybtex.database.BibliographyData`, so both
workflows expose their citations as tabular data rather than the software
side being a special string/BibTeX-object case. This does not change what a
user sees when calling `cite_software()` (still APA or BibTeX text) - it
changes the intermediate representation that both `render_apa()` and the
`fmt="bibtex"` path in `orchestrator.cite_software()` build from.

## Scope

In scope:
- `bibliography.collect_library_entries()` returns a `pandas.DataFrame`.
- Parsing switches from `pybtex` to `bibtexparser` (already an installed
  transitive dependency via `pylipd`/`pyleotups`/`doi2bib` - no new
  dependency).
- `render_apa()`, `generate_bibliography()`, and
  `orchestrator.cite_software()`'s `fmt="bibtex"` branch are updated to
  consume the DataFrame.
- A new `tests/test_bibliography.py` covering the DataFrame shape, DOI
  dedup, and citation-type filtering directly (today `collect_library_entries`
  is only exercised indirectly through `test_orchestrator.py`).

Out of scope (deferred to #27):
- `data_workflow.build_retrieval_cell()` discarding the metadata DataFrame
  that `get_bibtex()` (PyLiPD) and `get_publications()` (PyleoTUPS) already
  return as their second value. That DataFrame is a richer per-dataset
  metadata table (columns like `authors`, `doi`, `year`, `title`, `journal`,
  `citeKey`) built by those libraries before they ever construct BibTeX
  strings, and today it's discarded with `_bib_{variable}, _ = ...`. Real
  fix, but a separate function in a separate module - tracked as its own
  issue rather than folded in here.

## Architecture

```
bibliography.py
  collect_library_entries(libraries, citation_types=None) -> pd.DataFrame
      for each library: parse its "paper" (inline BibTeX string from
      library_citations.yml) and/or "software" (Citations/{lib}.bib) entries
      via bibtexparser, dedupe by DOI, append one row per entry.

  render_apa(entries: pd.DataFrame) -> str
      iterate entries["bibtex"], call llm.bibtex_to_apa() per row, join.

  generate_bibliography(...)
      unchanged except the truthiness check on collect_library_entries'
      result becomes `not entries.empty`.

orchestrator.py
  cite_software(...)
      fmt="bibtex": "\n\n".join(entries["bibtex"]) instead of
      entries.to_string(bib_format="bibtex")
      fmt="apa": unchanged call to render_apa(entries) - render_apa's
      signature change is transparent to this call site.
```

### DataFrame shape

One row per citation entry (a library with both a paper and a software
citation produces two rows). Columns:

| column | source | notes |
|---|---|---|
| `library` | loop variable | lowercased library name |
| `citation_type` | loop variable | `"paper"` or `"software"` |
| `key` | entry `ID` | the BibTeX cite key, e.g. `pyleoclim_software` |
| `title` | entry `title` | plain string, no LaTeX unescaping |
| `author` | entry `author` | plain string as `bibtexparser` gives it (not split into a list) |
| `year` | entry `year` | plain string |
| `doi` | entry `doi` | `""` when absent |
| `bibtex` | re-serialized via `bibtexparser`'s `BibTexWriter` | canonical single-entry BibTeX text, used by `render_apa` and the `fmt="bibtex"` path |

### DOI dedup

Same behavior as today: track `seen_dois`; skip an entry whose `doi` is
non-empty and already seen; entries without a DOI are never deduped against
each other (unchanged from the current `_add_entries` logic in
`bibliography.py`).

### Parsing library switch

`bibtexparser.loads(text).entries` / `bibtexparser.load(file).entries` return
plain `dict`s (not `pybtex` `Entry` objects with `Person`/field-list
structures), which map directly onto DataFrame columns - no manual
`Person`-to-string conversion needed. A small helper re-serializes a single
entry dict back to BibTeX text via `bibtexparser.bwriter.BibTexWriter` for
the `bibtex` column, since downstream (`bibtex_to_apa`) still expects a raw
BibTeX string per entry.

## Data flow summary

Unchanged at the edges: notebook -> imported libraries -> citations -> (APA).
Only the shape of "citations" in the middle changes, from
`pybtex.BibliographyData` to `pandas.DataFrame`.

## Error handling

No new error paths. Empty result (no libraries matched) is an empty
DataFrame with the same columns; `generate_bibliography`'s not-found
reporting is unaffected since it operates on `libraries`/`index` directly,
not on the collected entries.

## Testing

- `tests/test_bibliography.py` (new): DataFrame columns present, one row
  per paper/software entry, DOI dedup drops a second entry sharing a DOI,
  `citation_types` filtering keeps only the requested rows, `bibtex` column
  round-trips to a parseable BibTeX string.
- `tests/test_orchestrator.py` (existing): should pass unchanged since it
  asserts on substrings of `cite_software()`'s output text, not on the
  intermediate type. Run to confirm.

## Open questions

None - the data-workflow metadata-DataFrame gap is intentionally deferred to
its own issue rather than resolved here.
