# -*- coding: utf-8 -*-
"""Fetch initial conditions (ERA5 / GFS), run FourCastNet on the NIM, store steps + manifest.json."""
import logging

from dataiku.customrecipe import get_recipe_config

from earth2_dku.dku import params
from earth2_dku.forecast.runner import INPUTS_PREFIX, run_forecasts
from earth2_dku.storage.base import PrefixedStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("earth2.run")

config = get_recipe_config()
conn = params.connection_from_config(config)
model = params.build_model(conn)
source = params.build_source(config)
forecast_storage = params.folder_storage("forecast", output=True)
ic_storage = (params.folder_storage("initial_conditions", output=True, required=False)
              or PrefixedStorage(forecast_storage, INPUTS_PREFIX))

manifests = run_forecasts(
    params.build_client(conn, model), model, source, ic_storage, forecast_storage,
    params.resolve_init_times(config, source),
    params.as_int(config.get("simulation_length"), 6), params.parse_list(config.get("variables")),
    fail_on_partial=bool(config.get("fail_on_partial")),
)
for m in manifests:
    logger.info("Forecast %s: %s, %d steps", m.init_time, m.status, m.steps_received)
