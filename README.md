# NVIDIA Earth-2 FourCastNet plugin for Dataiku

Get AI weather forecasts for your own sites as a regular Dataiku dataset. The plugin sends the
current state of the atmosphere (ERA5 reanalysis or NOAA GFS) to an NVIDIA Earth-2 **FourCastNet NIM**,
stores the global forecast, and extracts values (for example 10 m wind speed every 6 hours at each
wind farm) into a table that you can join, chart, or feed to downstream models and dashboards.

| Component | Type | What it does |
|---|---|---|
| Earth-2 FourCastNet NIM connection | Preset | NIM endpoint, model version, timeout and GPU size |
| Earth-2: run forecast | Recipe | Downloads the ERA5 / GFS starting state, calls the NIM, stores the forecast steps → managed folder |
| Earth-2: extract to dataset | Recipe | Values at sites (points) or over a bounding box (region) → dataset |

Supported models (see `python-lib/earth2_dku/models/models.yaml`):

| Model id | NIM image | Input channels | Min. GPU memory |
|---|---|---|---|
| `fourcastnet-1.1.0` (default) | `nvcr.io/nim/nvidia/fourcastnet:1.1.0` (FourCastNet v2 SFNO) | 73 | 40 GB |
| `fourcastnet-2.0.0` | `nvcr.io/nim/nvidia/fourcastnet:2.0.0` (FourCastNet 3) | 72 | 60 GB |

## 1. Prerequisites

- A locally deployed FourCastNet NIM that DSS can reach over HTTP.
- NVIDIA AI Enterprise licensing for self-hosted NIMs.
- Outbound access from DSS to ERA5 (ARCO Zarr store on Google Cloud Storage) and/or NOAA GFS (AWS S3).
- A Python **3.11** or **3.12** code environment for the plugin. The code env installs Earth2Studio,
  which pulls in PyTorch, so it is large.

## 2. GPU guidance

- **1.1.0** runs on a single **L40S (48 GB)**.
- **2.0.0** needs a single GPU with **≥ 60 GB** (H100 80 GB or A100 80 GB). On an L40S it runs out of CUDA memory.
- A model can't be split across several GPUs. Pick the preset's model to match the GPU the NIM runs on.

## 3. Kubernetes deployment notes

- Deploy the NIM as a `NIMService` with a **plain PVC** for the model cache. **Don't use `NIMCache`**:
  the NIM Operator can't read profiles for non-LLM NIMs, and the cache fails with
  `no profiles are selected for caching`.
- Give the pod **≥ 4 GiB of shared memory** (`sharedMemorySizeLimit: 4Gi`). Otherwise the pod fails at startup.
- From DSS, use the **in-cluster service URL**, e.g.
  `http://fourcastnet.nim-services.svc.cluster.local:8000`. Port-forwards and load balancers have idle
  timeouts that cut long forecast streams.

## 4. Quickstart

1. **Preset.** In *Plugins → NVIDIA Earth-2 FourCastNet → Settings*, create an *Earth-2 FourCastNet NIM
   connection* preset with your NIM base URL and model `fourcastnet-1.1.0`. If the endpoint is
   protected by an API key (e.g. a NIM exposed through a gateway such as
   `https://<host>/v1/...` with `X-API-Key: <key>`), set *API key*, and *API key header* if your
   gateway uses a header other than `X-API-Key`. The key is sent with every request to the NIM.
2. **Run.** Add the *Earth-2: run forecast* recipe with a new output folder `forecast`, and select:
   - the preset
   - source *ERA5*, time mode *Fixed*, init time `2023-01-01T00:00:00Z`
   - simulation length `6` (36 h, 7 files including lead 0)
   - variables `u10m,v10m`

   The recipe checks that the NIM is up, downloads the initial condition, then streams the forecast.
   The `forecast` folder ends up with:

   ```
   inputs/era5_arco/fourcastnet-1.1.0/20230101T00.npy   initial condition, ~300 MB (cached)
   runs/init=2023-01-01T00/manifest.json
   runs/init=2023-01-01T00/steps/000_000.npy ... 036_000.npy
   ```

   To keep initial conditions apart (e.g. for a different retention policy), connect a second folder
   to the optional *Initial conditions folder* output. They are then cached there instead of under `inputs/`.
3. **Extract.** Upload `resource/sample_sites.csv` from this plugin as a dataset `sites`. Then add
   *Earth-2: extract to dataset* with inputs `forecast` and `sites`, and select:
   - mode *Points*
   - site columns `site_id`, `latitude`, `longitude`
   - derived `wind_speed_10m`

   The output `forecast_values` has one row per run, lead time, site and variable:

| column | type |
|---|---|
| run_init_time | datetime (UTC) |
| valid_time | datetime (UTC) |
| lead_hours | int |
| site_id | string |
| latitude / longitude | double |
| variable | string (`u10m`, `v10m`, `wind_speed_10m`, …) |
| value | double |
| model_id / source | string |

Choose layout *Wide* to get one column per variable instead. In *Region* mode, rows are grid cells
inside the bounding box (0.25°). `site_id` is then `"<lat>_<lon>"` and longitudes are given in −180..180.
A box with `lon_min > lon_max` crosses the antimeridian. Boxes around the 0° meridian
(e.g. −10..10) also work.

## 5. Scenario example: daily operational forecast

Set the run recipe to source *GFS* and time mode *Latest available*. The GFS safety margin defaults
to 6 h, to allow for publication delay. In the extract recipe, select *Latest run*.
Then create a scenario with a time-based trigger (e.g. daily at 07:00 UTC) and one step that builds
`forecast_values` recursively. Use run selection *All runs* if you want to keep a history for backtesting.

For backfills, connect a dataset with a timestamp column to the optional *Init times* input of the run
recipe. It runs one forecast per timestamp, each rounded down to a 00/06/12/18 UTC cycle.

## 6. How it works / design notes

- The two recipes only exchange data through the forecast folder. Its `manifest.json` records the
  model, the source, the init time, the step length, the **output variable order** and the step files.
  The extract recipe reads everything from the manifest and assumes no model.
- Only `u10m,v10m` are requested by default. All channels come to about 300 MB per step.
- NIM health is checked before anything is downloaded.
- Initial conditions are cached by `{source}/{model_id}/{time}` and reused on reruns. Their shape is
  checked (`(1, 1, n_channels, 721, 1440)`, float32) before they are cached or sent to the NIM.
- If the NIM stream breaks, the steps received so far are kept and the manifest says
  `status: "partial"`. Unless *Fail on partial forecast* is checked, the run recipe then succeeds with a
  warning (if at least one step arrived). The extract recipe uses partial runs and logs a warning.
- **`variables` parameter on 1.1.0:** it is not yet verified that NIM 1.1.0 honours it
  (`supports_variables_param: unverified`). The client sends it, and if the NIM rejects it, retries once
  without it. Run the live variable-order test against your NIM (see *Development*) and record the
  result in `models.yaml` and here.

## 7. Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| `size of tensor a (73) must match ... (72)` in NIM logs; the recipe reports a broken stream | Input variable list doesn't match the model version | Check the preset's model matches the deployed NIM image. The plugin checks the input shape against the preset's model before sending |
| `CUDA out of memory` in NIM logs | 2.0.0 on a 48 GB GPU | Use 1.1.0 on L40S, or an 80 GB GPU |
| NIM Operator `NIMCache`: `no profiles are selected for caching` | The operator can't read profiles for non-LLM NIMs | Deploy `NIMService` with a plain PVC, no `NIMCache` |
| HTTP 400/500 on `/v1/infer` | Time format or array shape | The plugin sends RFC 3339 without fractional seconds and validates the shape. Check the NIM logs |
| Slow or very large responses | ~300 MB per step with all 73 channels | Keep `variables=u10m,v10m`; use the region cell limit |
| Pod fails at startup | Shared memory too small | `sharedMemorySizeLimit` ≥ 4 GiB |
| Health check returns HTTP 401/403 | The endpoint requires an API key | Set *API key* (and *API key header*) in the preset |
| "The NIM stream broke after N step(s)" | Server-side failure after HTTP 200, or an idle timeout on a proxy | Check the NIM pod logs; use the in-cluster service URL |

## Development

```
python-lib/earth2_dku/   pure-Python core (only dku/ and storage/dku_folder.py import dataiku)
tests/python/unit/       unit tests: no DSS, no network, no Earth2Studio
tests/python/live/       NIM / ERA5 / GFS integration tests (markers `nim`, `network`)
```

- `make unit-tests` runs the unit tests in a fresh venv.
- `EARTH2_NIM_URL=http://host:8000 make live-tests` runs the live tests. These need the full code env.
  For a protected endpoint, also set `EARTH2_NIM_API_KEY` (and `EARTH2_NIM_API_KEY_HEADER` if not
  `X-API-Key`). Optionally, set `EARTH2_MODEL_ID` and `EARTH2_INPUT_NPY` (a pre-fetched initial condition) to skip the
  ERA5 download.
- `make plugin` builds the zip for *Plugins → Add plugin → Upload*.

To add a model version, add an entry to `models.yaml` and a choice to the preset's `model_id` (a unit
test checks that the two stay in sync).
