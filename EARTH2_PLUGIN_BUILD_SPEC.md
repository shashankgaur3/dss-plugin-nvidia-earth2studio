# Build Spec: `dss-plugin-nvidia-earth2` (Dataiku plugin for NVIDIA Earth-2 FourCastNet NIM)

You are building a Dataiku DSS plugin that lets users run NVIDIA Earth-2 FourCastNet weather forecasts (served as a NIM) and turn the output into regular Dataiku datasets for business use (e.g. wind forecasts at wind-farm sites).

Read this whole file before writing code. Follow the architecture and build order exactly. If something here conflicts with the actual Dataiku plugin schema or the NIM API you observe, stop and report it instead of silently improvising.

---

## 1. Goals and non-goals

**Phase 1 (build this):**
- A connection preset for a FourCastNet NIM endpoint.
- Three custom recipes: fetch initial conditions, run forecast, extract to dataset.
- One macro: preflight check.
- A model registry with FourCastNet 1.1.0 (primary) and 2.0.0.
- A pure-Python core library with unit tests that run without DSS.

**Not in phase 1 (do not build, but do not design them out):**
- Map webapp, backtest recipe, ensemble runs, CorrDiff or other Earth-2 models, a generic multi-NIM framework.

---

## 2. Architecture rules (non-negotiable)

1. **Thin Dataiku shell.** Every `recipe.py` and `runnable.py` only: reads config, builds core objects via `earth2_dku/dku/params.py`, calls one core function, writes outputs. Target ~30-50 lines each. No weather/array logic in recipe files.
2. **Only `earth2_dku/dku/` and `earth2_dku/storage/dku_folder.py` may `import dataiku`.** Everything else must be importable and testable outside DSS.
3. **Recipes communicate only through the folder layout + `manifest.json`** (section 6). The extract recipe must never assume a model, variable order, or step length; it reads them from the manifest.
4. **Model-specific facts live in `models.yaml`**, not in code branches. Adding a model version should mean adding a registry entry.
5. **Interfaces before implementations.** Define the abstract base classes in section 5 first, then implement.

---

## 3. Repository layout

```
dss-plugin-nvidia-earth2/
├── plugin.json
├── code-env/python/
│   ├── desc.json
│   └── spec/requirements.txt
├── parameter-sets/
│   └── earth2-nim-connection/parameter-set.json
├── custom-recipes/
│   ├── earth2-fetch-initial-conditions/{recipe.json, recipe.py}
│   ├── earth2-run-forecast/{recipe.json, recipe.py}
│   └── earth2-extract-to-dataset/{recipe.json, recipe.py}
├── python-runnables/
│   └── earth2-preflight-check/{runnable.json, runnable.py}
├── resource/
│   └── sample_sites.csv
├── python-lib/earth2_dku/
│   ├── __init__.py
│   ├── errors.py
│   ├── models/{__init__.py, registry.py, models.yaml}
│   ├── clients/{__init__.py, base.py, self_hosted.py, nvcf_hosted.py}
│   ├── sources/{__init__.py, base.py, era5_arco.py, gfs.py, cache.py}
│   ├── forecast/{__init__.py, runner.py, tar_stream.py, manifest.py}
│   ├── extract/{__init__.py, grid.py, points.py, region.py, derived.py}
│   ├── storage/{__init__.py, base.py, local.py, dku_folder.py}
│   ├── validation/{__init__.py, preflight.py}
│   └── dku/{__init__.py, params.py}
├── tests/
│   ├── conftest.py
│   ├── fakes.py            # FakeNimClient, FakeSource, synthetic tar builder
│   └── test_*.py
└── README.md
```

Plugin ID: `nvidia-earth2`. Python package: `earth2_dku`.

---

## 4. Code environment

`code-env/python/desc.json`:
```json
{
  "acceptedPythonInterpreters": ["PYTHON311", "PYTHON310"],
  "forceConda": false,
  "installCorePackages": true,
  "installJupyterSupport": false
}
```

`spec/requirements.txt`: `earth2studio>=0.3.0`, `numpy`, `xarray`, `requests`, `pyyaml`, `pandas`, `scipy` (for interpolation).

Notes:
- NVIDIA documents Earth2Studio for Python >=3.10,<3.12. Prefer 3.11. (A 3.12 env has worked in testing, but don't rely on it.)
- Earth2Studio is heavy. Import it **lazily** inside `sources/era5_arco.py` and `sources/gfs.py` only, never at package import time, so the run/extract recipes and tests don't pay for it.

---

## 5. Core library: modules and interfaces

### 5.1 `errors.py`
Define `Earth2Error` and subclasses: `ConfigError`, `PreflightError`, `NimRequestError`, `NimStreamBrokenError` (carries `steps_received: int`), `HostedLargeAssetError`, `ManifestError`. Every error message must say what the user should do next.

### 5.2 `models/models.yaml` and `registry.py`

```yaml
fourcastnet-1.1.0:
  label: "FourCastNet v2 SFNO (NIM 1.1.0)"
  nim_image: "nvcr.io/nim/nvidia/fourcastnet:1.1.0"
  input_variables_from: "earth2studio.models.px.sfno:VARIABLES"
  n_channels: 73
  step_hours: 6
  min_gpu_mem_gb: 40
  supports_variables_param: unverified   # true | false | unverified
  grid: era5_0p25

fourcastnet-2.0.0:
  label: "FourCastNet 3 (NIM 2.0.0)"
  nim_image: "nvcr.io/nim/nvidia/fourcastnet:2.0.0"
  input_variables_from: "earth2studio.models.px.fcn3:VARIABLES"
  n_channels: 72
  step_hours: 6
  min_gpu_mem_gb: 60
  supports_variables_param: true
  grid: era5_0p25
```

`registry.py`:
- `ModelSpec` dataclass mirroring the YAML.
- `get_model(model_id) -> ModelSpec`, `list_models() -> list[ModelSpec]`.
- `ModelSpec.input_variables() -> list[str]`: lazily imports the module path in `input_variables_from` and asserts `len(...) == n_channels`. Raise `ConfigError` on mismatch (this catches Earth2Studio version drift).

### 5.3 `clients/base.py`

```python
class NimClient(ABC):
    def health(self) -> HealthStatus: ...          # ready: bool, detail: str
    def infer_stream(
        self, input_array_bytes: bytes, init_time: datetime,
        simulation_length: int, variables: list[str] | None,
    ) -> Iterator[ForecastStep]: ...               # yields steps as they arrive

@dataclass
class ForecastStep:
    lead_hours: int
    batch_idx: int
    filename: str          # e.g. "006_000.npy"
    raw_bytes: bytes       # the .npy bytes, unchanged
```

### 5.4 `clients/self_hosted.py`
- `POST {base_url}/v1/infer`, multipart: `input_array` file + form fields `input_time`, `simulation_length`, optional `variables` (comma-separated). Header `accept: application/x-tar`. `stream=True`.
- `input_time` must be RFC 3339 **without fractional seconds**: `%Y-%m-%dT%H:%M:%SZ`.
- Parse the response with `tarfile.open(fileobj=resp.raw, mode="r|")` (streaming mode). Filenames are `{lead}_{batch}.npy`; `lead` is in hours in observed output (000, 006, 012...). Parse both numbers from the stem.
- If the stream breaks mid-way (`requests.exceptions.ChunkedEncodingError`, `tarfile.ReadError`, urllib3 `ProtocolError`), raise `NimStreamBrokenError(steps_received=n)` with a message telling the user to check the NIM pod logs. **Background:** the NIM returns HTTP 200 and starts streaming before inference finishes, so server-side failures (wrong channel count, CUDA OOM) surface only as a truncated stream.
- If `supports_variables_param` is `false`, never send `variables`. If `unverified` and the server returns 400/422 mentioning `variables`, retry once without it and log a warning.
- `health()`: `GET /v1/health/ready`, 200 means ready.
- Configurable timeout (default 1200 s).

### 5.5 `clients/nvcf_hosted.py` (experimental, hosted API)
- Endpoint `https://climate.api.nvidia.com/v1/nvidia/fourcastnet`, JSON body (`input_id`, `variables`, `simulation_length`, `ensemble_size`, `noise_amplitude`), Bearer auth, header `NVCF-POLL-SECONDS: 300`.
- Handle: 202 then poll `https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{NVCF-REQID}` with `allow_redirects=False`; 302 then GET `Location` **without** auth header.
- **Known issue:** large results may return HTTP 200 with body `{"message":"Large asset written"}` and no retrievable URL (status endpoint returns 410). Detect this body and raise `HostedLargeAssetError` advising a smaller `simulation_length` or self-hosting. Never write that body to disk as if it were output.
- The hosted API returns a zip/tar rather than a stream; adapt it into `ForecastStep`s by unpacking. Sniff magic bytes (`PK\x03\x04` zip, `\x1f\x8b` gzip, `ustar` at offset 257 tar).
- Mark this client "experimental" in the UI label.

### 5.6 `sources/base.py`, `era5_arco.py`, `gfs.py`, `cache.py`

```python
class InitialConditionSource(ABC):
    name: str
    def get_state(self, time: datetime, variables: list[str]) -> np.ndarray: ...
    # returns float32, shape (1, 1, n_vars, 721, 1440)
    def latest_available_time(self) -> datetime: ...
```
- `era5_arco.py`: `earth2studio.data.ARCO()(time=..., variable=...)`, then `.to_numpy()[None].astype("float32")`. ERA5 is historical; the preliminary version lags ~5 days. `latest_available_time()` returns now minus a conservative lag rounded down to 00/06/12/18 UTC.
- `gfs.py`: `earth2studio.data.GFS`. Cycles every 6 h; `latest_available_time()` returns the most recent cycle minus a safety margin (configurable, default 6 h).
- `cache.py`: `CachedSource(source, storage)` wraps any source; key = `f"{source.name}/{model_id}/{time:%Y%m%dT%H}.npy"`. Reuse if present.
- After building, assert `arr.shape == (1, 1, model.n_channels, 721, 1440)` and dtype float32, else `PreflightError`.

### 5.7 `storage/`

```python
class Storage(ABC):
    def exists(self, path: str) -> bool: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def list(self, prefix: str = "") -> list[str]: ...
```
- `local.py`: filesystem-backed, for tests and notebooks.
- `dku_folder.py`: wraps `dataiku.Folder`; use `get_writer` / `get_download_stream` / `list_paths_in_partition`. **Normalize paths**: DSS returns paths with a leading `/`; strip/add consistently so `exists()` is reliable.

### 5.8 `forecast/manifest.py`, `tar_stream.py`, `runner.py`
- `manifest.py`: dataclass + JSON (de)serialization for the schema in section 6. Validate on read.
- `runner.py`:
  ```python
  def run_forecast(client, model, storage, run_dir, input_bytes, init_time,
                   simulation_length, variables) -> Manifest
  ```
  Writes `manifest.json` with `status="running"` first, writes each step to `steps/` as it arrives, updates the manifest at the end (`status="complete"` or `"partial"` with `steps_received`), and re-raises errors after recording state. Partial results must be preserved.

### 5.9 `extract/`
- `grid.py`: ERA5 0.25° grid. Latitudes `90 → -90` (721 values), longitudes `0 → 359.75` (1440 values). `to_grid_lon(lon)` converts -180..180 to 0..360. Index lookup helpers.
- `points.py`: `extract_points(manifest, storage, sites_df, variables, method="nearest"|"bilinear") -> pd.DataFrame`. Handle longitude wraparound for bilinear at 359.75 to 0.
- `region.py`: bounding-box subset (handle boxes crossing the 0° meridian) returning a long-format DataFrame. Warn if the cell count exceeds a configurable limit (default 50,000 cells per step).
- `derived.py`: a registry `@derived("wind_speed_10m", requires=["u10m","v10m"])`. Phase 1 entries: `wind_speed_10m`, `wind_speed_100m` (u100m/v100m), `t2m_celsius`. Unknown names raise `ConfigError` listing the available ones.
- **Array shape:** each step `.npy` is observed as `(batch, time, n_vars, 721, 1440)`. Do not hardcode; reshape with `arr.reshape(-1, 721, 1440)` after asserting the product of leading dims equals the manifest's variable count.
- **Variable order:** take it from `manifest.output_variables`. That is the requested `variables` list if one was sent, else the model's full input variable list. **This ordering assumption must be verified with an integration test** (section 10): request `u10m,v10m` in both orders and confirm the fields swap.

Output dataset schema (long format, default):

| column | type |
|---|---|
| run_init_time | datetime (UTC) |
| valid_time | datetime (UTC) |
| lead_hours | int |
| site_id | string |
| latitude | double |
| longitude | double |
| variable | string |
| value | double |
| model_id | string |
| source | string |

Wide format option: one column per variable, same keys.

### 5.10 `validation/preflight.py`
Checks, each returning `(name, ok, message)`:
1. Endpoint health is ready.
2. Selected model exists in the registry.
3. `input_variables()` length matches `n_channels` for the installed Earth2Studio.
4. Optional: user-supplied GPU memory (GB) >= `min_gpu_mem_gb`. Message example: "FourCastNet NIM 2.0.0 needs a single GPU with 60 GB; an L40S (48 GB) will run out of memory. Use 1.1.0 on L40S, or an 80 GB GPU. The model cannot be split across GPUs."
5. Init time format and availability for the selected source.

### 5.11 `dku/params.py`
Translate recipe/preset config dicts into `ModelSpec`, `NimClient`, `InitialConditionSource`, `Storage`. All Dataiku-specific config parsing lives here.

---

## 6. Data contract: folder layout and manifest

```
<forecast folder>/
  runs/init=2023-01-01T00/
    manifest.json
    steps/000_000.npy
    steps/006_000.npy
    ...
<initial conditions folder>/
  era5_arco/fourcastnet-1.1.0/20230101T00.npy
```

`manifest.json`:
```json
{
  "schema_version": 1,
  "model_id": "fourcastnet-1.1.0",
  "nim_image": "nvcr.io/nim/nvidia/fourcastnet:1.1.0",
  "source": "era5_arco",
  "init_time": "2023-01-01T00:00:00Z",
  "step_hours": 6,
  "simulation_length": 6,
  "output_variables": ["u10m", "v10m"],
  "grid": {"name": "era5_0p25", "n_lat": 721, "n_lon": 1440,
           "lat_start": 90.0, "lat_step": -0.25, "lon_start": 0.0, "lon_step": 0.25},
  "steps": [{"file": "steps/000_000.npy", "lead_hours": 0, "batch_idx": 0}],
  "status": "complete",
  "steps_received": 7,
  "created_at": "2026-11-24T00:00:00Z",
  "plugin_version": "0.1.0"
}
```
Note: `simulation_length = N` produces N+1 files (lead 0 is the initial state).

---

## 7. Dataiku components

Verify every JSON file below against the current Dataiku plugin documentation or a `dataiku plugin` scaffold before finalizing; field names here are the intended shape.

### 7.1 Preset `parameter-sets/earth2-nim-connection/parameter-set.json`
Params:
- `deployment` SELECT: `self_hosted` (default) | `nvcf_hosted` (label "NVIDIA hosted API (experimental)").
- `base_url` STRING (self-hosted), e.g. `http://fourcastnet.nim-services.svc.cluster.local:8000`.
- `api_key` PASSWORD (hosted only; `visibilityCondition` on deployment).
- `model_id` SELECT from the registry IDs, default `fourcastnet-1.1.0`.
- `timeout_seconds` INT, default 1200.
- `gpu_memory_gb` INT, optional, used by preflight.

Definable inline and at project level.

### 7.2 Recipe `earth2-fetch-initial-conditions`
- Inputs: none required (optional dataset role `dates` with a timestamp column, for batch backfills).
- Output: managed folder role `initial_conditions`.
- Params: preset; `source` SELECT `era5_arco` | `gfs`; `time_mode` SELECT `fixed` | `latest_available` | `project_variable`; `init_time` STRING (fixed); `project_variable_name` STRING.
- Behavior: resolve time, round to a 6-hour cycle, fetch via `CachedSource`, validate shape.

### 7.3 Recipe `earth2-run-forecast`
- Input: managed folder `initial_conditions`. Output: managed folder `forecast`.
- Params: preset; `init_time` (default: latest file present in the input folder); `simulation_length` INT default 6; `variables` STRING comma-separated, default `u10m,v10m`; `fail_on_partial` BOOLEAN default false.
- Behavior: preflight (health + shape) then `run_forecast`. On a broken stream, keep partial output, set the manifest status, and fail with a message pointing to pod logs unless `fail_on_partial` is false and at least one step arrived.

### 7.4 Recipe `earth2-extract-to-dataset`
- Inputs: managed folder `forecast` (required); dataset `sites` (optional).
- Output: dataset `forecast_values`.
- Params: `run_selection` SELECT `latest` | `all` | `specific` (+ `init_time`); `mode` SELECT `points` | `region`; `site_id_column`, `lat_column`, `lon_column` (COLUMN params on `sites`); bbox `lat_min`, `lat_max`, `lon_min`, `lon_max`; `variables` (blank = all in manifest); `derived` MULTISELECT from the derived registry; `interpolation` SELECT `nearest` | `bilinear`; `layout` SELECT `long` | `wide`.
- Behavior: read manifest(s), extract, write the dataset with an explicit schema (section 5.9).

### 7.5 Macro `earth2-preflight-check`
- Params: preset.
- Runs all `preflight.py` checks; returns an HTML table (check, status, message).

### 7.6 `resource/sample_sites.csv`
5 to 10 illustrative wind-farm-style sites with columns `site_id, name, latitude, longitude` (use clearly generic names, e.g. "Site A - West Texas").

---

## 8. Known pitfalls (encode these in code and error messages)

| Symptom | Cause | Handling |
|---|---|---|
| `size of tensor a (73) must match ... (72)` in NIM logs, client sees `ChunkedEncodingError` | Wrong input variable list for the model version | Registry + shape assert before sending |
| `CUDA out of memory` in NIM logs | 2.0.0 on a 48 GB GPU | Preflight GPU check; message suggests 1.1.0 or 80 GB GPU |
| NIM Operator NIMCache: `no profiles are selected for caching` | Operator can't read profiles for non-LLM NIMs | README: deploy NIMService with a plain PVC, no NIMCache |
| Hosted API returns `{"message":"Large asset written"}` | Hosted large-output handoff is broken | `HostedLargeAssetError` |
| 400/500 on `/v1/infer` | Time format or shape | Enforce RFC 3339 without fractional seconds; validate shape |
| Slow/large responses | ~300 MB per step for all 73 channels | Default `variables=u10m,v10m`; region cell limit |
| Pod fails at startup | Shared memory | README: shm >= 4 GiB (`sharedMemorySizeLimit`) |

---

## 9. Build order

Do these in order. Finish tests for each step before moving on.

1. Repo skeleton, `plugin.json`, code env files, empty packages.
2. `errors.py`, `storage/base.py` + `local.py`, `models/` registry + YAML. Tests.
3. `forecast/manifest.py`. Tests (round-trip, validation errors).
4. `clients/base.py`, `self_hosted.py` with `tests/fakes.py` (an in-process fake that serves a synthetic tar stream, including a mid-stream break). Tests.
5. `forecast/runner.py`. Tests: complete run, partial run preserved.
6. `extract/grid.py`, `points.py`, `region.py`, `derived.py`. Tests with synthetic arrays where values encode their own lat/lon so correctness is checkable; include longitude wraparound and the 0° meridian.
7. `sources/` with lazy imports; unit-test `cache.py` with `FakeSource`. Mark real ERA5/GFS fetches as integration tests.
8. `validation/preflight.py`. Tests.
9. `dku/params.py`, `storage/dku_folder.py`.
10. Recipes, preset, macro JSON + thin Python.
11. `clients/nvcf_hosted.py` (experimental). Tests with fakes for 202/302/large-asset cases.
12. README.

---

## 10. Testing

- `pytest`, no DSS and no network required for unit tests.
- Integration tests behind markers: `@pytest.mark.nim` (needs `EARTH2_NIM_URL`), `@pytest.mark.network` (ERA5/GFS fetch).
- Required integration tests:
  - Health check passes.
  - `simulation_length=1` returns 2 steps with expected shapes.
  - **Variable-order test:** request `u10m,v10m` and `v10m,u10m`; assert the two fields swap. Record the result in `models.yaml` (`supports_variables_param` for 1.1.0) and in the README.
  - Point extraction on a real step returns plausible 10 m wind values (|u|,|v| < 100 m/s).

---

## 11. README contents

1. What the plugin does (one paragraph, business framing: forecasts at your sites as a dataset).
2. Prerequisites: a FourCastNet NIM reachable from DSS; NVIDIA AI Enterprise licensing for self-hosted NIMs; outbound access to ERA5 (ARCO on Google Cloud) and/or GFS.
3. GPU guidance: 1.1.0 runs on one L40S (48 GB); 2.0.0 needs a single GPU with >= 60 GB (H100/A100 80 GB). Models cannot be split across GPUs.
4. Kubernetes deployment notes: NIMService with plain PVC (no NIMCache), shm >= 4 GiB, in-cluster service URL preferred over port-forward or load balancers (idle timeouts break long streams).
5. Quickstart flow: preset, fetch (ERA5, 2023-01-01T00), run forecast (length 6, `u10m,v10m`), extract with `sample_sites.csv` and derived `wind_speed_10m`.
6. Scenario example: daily run with `source=gfs`, `time_mode=latest_available`.
7. Troubleshooting: section 8 table.

---

## 12. Definition of done (phase 1)

- All unit tests pass without DSS or network.
- Plugin installs in DSS from a zip; code env builds.
- Quickstart flow in README runs end to end against a FourCastNet 1.1.0 NIM on an L40S and produces a `forecast_values` dataset with `wind_speed_10m` per site per lead time.
- Preflight macro correctly flags: endpoint down, 2.0.0 configured with `gpu_memory_gb=48`.
- No recipe file exceeds ~60 lines; no module outside `dku/` and `storage/dku_folder.py` imports `dataiku`.
