"""
Tests for deterministic, source-oriented dataset detection.

The detector is tested through its notebook-path entry point. These fixtures
exercise static data-flow boundaries without importing or executing the
scientific libraries themselves.
"""

import inspect
import os
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deterministic_dataset_detection import (
    detect_datasets_in_notebook,
    detect_datasets_with_diagnostics,
)


def _write_notebook(path, code):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(code)]
    )
    with open(path, "w") as handle:
        nbformat.write(notebook, handle)


def test_public_entry_point_accepts_only_notebook_path():
    assert list(inspect.signature(detect_datasets_in_notebook).parameters) == [
        "notebook_path"
    ]


def test_diagnostics_warn_when_analysis_has_unrecognized_source(tmp_path):
    notebook = tmp_path / "unknown_loader.ipynb"
    _write_notebook(
        notebook,
        """
import custom_loader

df = custom_loader.load_data("remote://example")
result = df.pca()
""",
    )

    diagnostics = detect_datasets_with_diagnostics(str(notebook))

    assert diagnostics["pairs"] == []
    assert len(diagnostics["warnings"]) == 1
    assert "pca" in diagnostics["warnings"][0]
    assert "unsupported loader" in diagnostics["warnings"][0]


def test_diagnostics_warn_when_recognized_source_is_not_activated(tmp_path):
    notebook = tmp_path / "unrecognized_pyleotups_loader.ipynb"
    _write_notebook(
        notebook,
        """
from pyleotups import PangaeaDataset

ds = PangaeaDataset()
ds.fetch_studies()
data = ds.get_data()
result = data.pca()
""",
    )

    diagnostics = detect_datasets_with_diagnostics(str(notebook))

    assert diagnostics["pairs"] == []
    assert len(diagnostics["warnings"]) == 1
    assert "PyleoTUPS" in diagnostics["warnings"][0]
    assert "activated" in diagnostics["warnings"][0]


def test_path_only_entry_point_detects_lipdgraph_terminal_dataframe(tmp_path):
    notebook = tmp_path / "paleoPCAlite.ipynb"
    _write_notebook(
        notebook,
        """
import io
import pandas as pd
import pyleoclim as pyleo
import requests

url = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"
response = requests.post(url, data={"query": "SELECT ..."})
df_res = pd.read_csv(io.StringIO(response.text))
df = df_res[df_res["varID"].notna()]
filtered_df = df[df["timeval"].notna()]
filtered_df2 = filtered_df[filtered_df["varname"].notna()]
ts_list = []
for _, row in filtered_df2.iterrows():
    ts_list.append(pyleo.GeoSeries(time=row["timeval"], value=row["val"]))
mgs = pyleo.MultipleGeoSeries(ts_list)
mgs_common = mgs.common_time()
pca = mgs_common.pca()
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [
        ["filtered_df2", "LiPDGraph"]
    ]


def test_no_analysis_uses_latest_filtered_table_boundary(tmp_path):
    notebook = tmp_path / "filtered_only.ipynb"
    _write_notebook(
        notebook,
        """
import pandas as pd

raw = pd.read_csv("records.csv")
filtered = raw[raw["value"].notna()]
final = filtered.drop_duplicates(subset=["id"]).reset_index(drop=True)
display(final)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["final", "pandas"]]


def test_terminal_fallback_is_per_source_alongside_resolved_analysis(tmp_path):
    notebook = tmp_path / "mixed_sources.ipynb"
    _write_notebook(
        notebook,
        """
import pandas as pd
import xarray as xr

main = xr.open_dataset("main.nc")
solver = Eof(main)
scratch = pd.read_csv("lookup.csv")
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [
        ["main", "xarray"],
        ["scratch", "pandas"],
    ]


def test_bare_merge_does_not_promote_a_pylipd_source(tmp_path):
    notebook = tmp_path / "bare_merge.ipynb"
    _write_notebook(
        notebook,
        """
from pylipd.lipd import LiPD
from mylib import merge

D = LiPD()
D.load("x.lpd")
out = merge(D, 1)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == []


def test_bare_dataframe_does_not_promote_a_pylipd_source(tmp_path):
    notebook = tmp_path / "bare_dataframe.ipynb"
    _write_notebook(
        notebook,
        """
from pylipd.lipd import LiPD
from someorm import DataFrame

D = LiPD()
D.load("x.lpd")
out = DataFrame(D)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == []


def test_sparql_dataframe_helpers_produce_one_result_per_query(tmp_path):
    notebook = tmp_path / "sparql_helpers.ipynb"
    _write_notebook(
        notebook,
        """
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

def fetch_sparql(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return pd.DataFrame(results["results"]["bindings"])

endpoint = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"
first = fetch_sparql(endpoint, "first")
first_filtered = first.drop_duplicates(subset=["id"])
second = fetch_sparql(endpoint, "second")
second_filtered = second[second["value"].notna()]
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [
        ["first_filtered", "LiPDGraph"],
        ["second_filtered", "LiPDGraph"],
    ]


def test_direct_sparqlwrapper_dataframe_is_a_lipdgraph_source(tmp_path):
    notebook = tmp_path / "direct_sparql.ipynb"
    _write_notebook(
        notebook,
        """
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

endpoint = "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic"
sparql = SPARQLWrapper(endpoint)
sparql.setReturnFormat(JSON)
result = sparql.query().convert()
final = pd.DataFrame(result["results"]["bindings"])
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["final", "LiPDGraph"]]


def test_pyleotups_get_data_with_explicit_study_id_activates_source(tmp_path):
    notebook = tmp_path / "direct_pyleotups_data.ipynb"
    _write_notebook(
        notebook,
        """
import pyleotups as pt

ds = pt.PangaeaDataset()
data = ds.get_data(study_id="830587")[0]
fit = SomeModel().fit(data)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["ds", "PyleoTUPS"]]


def test_unused_pylipd_load_is_omitted_without_analysis(tmp_path):
    notebook = tmp_path / "unused_lipd.ipynb"
    _write_notebook(
        notebook,
        """
from pylipd.lipd import LiPD

D = LiPD()
D.load_remote_datasets(["TR04EVLI"])
names = D.get_all_dataset_names()
print(names)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == []


def test_pylipd_object_reaching_analysis_is_reported_as_source_object(tmp_path):
    notebook = tmp_path / "used_lipd.ipynb"
    _write_notebook(
        notebook,
        """
from pylipd.lipd import LiPD
import pyleoclim as pyleo

D = LiPD()
D.load("dataset.lpd")
timeseries = D.get_timeseries(D.get_all_dataset_names())
series = pyleo.Series(time=timeseries.time, value=timeseries.value)
result = series.pca()
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["D", "PyLiPD"]]


def test_pyleotups_search_without_analysis_is_omitted(tmp_path):
    notebook = tmp_path / "search_only.ipynb"
    _write_notebook(
        notebook,
        """
import pyleotups as pt

ds = pt.PangaeaDataset()
results = ds.search_studies(study_ids=830587)
bib, metadata = ds.get_publications()
display(metadata)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == []


def test_pyleotups_get_data_reaching_fit_reports_source_object(tmp_path):
    notebook = tmp_path / "pyleotups_analysis.ipynb"
    _write_notebook(
        notebook,
        """
import pyleotups as pt

ds = pt.PangaeaDataset()
ds.search_studies(study_ids=830587)
data = ds.get_data()
model = SomeModel()
fit = model.fit(data)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["ds", "PyleoTUPS"]]


def test_xarray_dataset_boundary_reaching_eof_is_reported(tmp_path):
    notebook = tmp_path / "xarray_analysis.ipynb"
    _write_notebook(
        notebook,
        """
import xarray as xr
from eofs.xarray import Eof

ds = xr.open_dataset("model.nc")
ds_geo_time = ds.sel(time=slice("0910", "1642"))
ds_geo = ds_geo_time.resample(time="20A").mean()
signal = ds_geo["precip"] - ds_geo["precip"].mean(dim="time")
solver = Eof(signal)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["ds_geo", "xarray"]]


def test_inspection_and_plotting_methods_report_terminal_table(tmp_path):
    notebook = tmp_path / "inspection_only.ipynb"
    _write_notebook(
        notebook,
        """
import io
import pandas as pd
import requests

response = requests.post(
    "https://linkedearth.graphdb.mint.isi.edu/repositories/LiPDVerse-dynamic",
    data={"query": "SELECT ..."},
)
df = pd.read_csv(io.StringIO(response.text))
df.head()
df.info()
df.describe()
df.plot()
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["df", "LiPDGraph"]]


def test_merged_pyleotups_objects_report_terminal_merged_name(tmp_path):
    notebook = tmp_path / "merged_pyleotups.ipynb"
    _write_notebook(
        notebook,
        """
import pyleotups as pt

ds1 = pt.PangaeaDataset()
ds1.search_studies(study_ids=830587)
ds2 = pt.NOAADataset()
ds2.search_studies(noaa_id=33213)
ds_sum = ds1 + ds2
model = SomeModel()
fit = model.fit(ds_sum)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == [["ds_sum", "PyleoTUPS"]]


def test_results_are_deterministic_and_source_ordered(tmp_path):
    notebook = tmp_path / "ordered.ipynb"
    _write_notebook(
        notebook,
        """
import pandas as pd
import xarray as xr

table = pd.read_csv("table.csv")
grid = xr.open_dataset("grid.nc")
model = SomeModel()
grid_fit = model.fit(grid)
table_fit = model.fit(table)
""",
    )

    expected = [["table", "pandas"], ["grid", "xarray"]]
    assert detect_datasets_in_notebook(str(notebook)) == expected
    assert detect_datasets_in_notebook(str(notebook)) == expected


def test_generated_provenance_cells_are_ignored(tmp_path):
    notebook = tmp_path / "generated.ipynb"
    _write_notebook(
        notebook,
        """
# provenance-agent-generated
from pylipd.lipd import LiPD
_lipd_D = LiPD()
_lipd_D.load_remote_datasets(["TR04EVLI"])
_bib_D, _meta_D = _lipd_D.get_bibtex(remote=True)
display(_meta_D)
""",
    )

    assert detect_datasets_in_notebook(str(notebook)) == []


def test_repository_paleo_pca_lite_fixture_matches_terminal_expectation():
    repository_root = Path(__file__).resolve().parents[1]
    notebook = repository_root / "notebooks/testing/paleoPCAlite.ipynb"

    assert detect_datasets_in_notebook(str(notebook)) == [
        ["filtered_df2", "LiPDGraph"]
    ]


def test_repository_paleo_pca_fixture_reports_both_analysis_sources():
    repository_root = Path(__file__).resolve().parents[1]
    notebook = repository_root / "notebooks/testing/paleoPCA.ipynb"

    assert detect_datasets_in_notebook(str(notebook)) == [
        ["filtered_df2", "LiPDGraph"],
        ["ds_geo", "xarray"],
    ]
