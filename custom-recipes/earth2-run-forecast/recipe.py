# -*- coding: utf-8 -*-
"""Run a FourCastNet forecast on the NIM; store steps + manifest.json in the forecast folder."""
import logging

from dataiku.customrecipe import get_recipe_config

from earth2_dku.dku import params
from earth2_dku.errors import NimStreamBrokenError
from earth2_dku.forecast.manifest import run_dir_for
from earth2_dku.forecast.runner import run_forecast
from earth2_dku.sources.cache import find_initial_condition
from earth2_dku.validation.preflight import ensure_ready_for_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("earth2.run")

config = get_recipe_config()
conn = params.connection_from_config(config)
model = params.build_model(conn)
client = params.build_client(conn, model)
ic_storage = params.folder_storage("initial_conditions")
forecast_storage = params.folder_storage("forecast", output=True)

ic = find_initial_condition(ic_storage, model.model_id, params.optional_time(config, "init_time"))
input_bytes = ic_storage.read_bytes(ic.path)
ensure_ready_for_forecast(client, model, input_bytes)

try:
    manifest = run_forecast(
        client, model, forecast_storage, run_dir_for(ic.time), input_bytes, ic.time,
        params.as_int(config.get("simulation_length"), 6), params.parse_list(config.get("variables")),
        source=ic.source,
    )
    logger.info("Forecast complete: %d steps in %s", manifest.steps_received, run_dir_for(ic.time))
except NimStreamBrokenError as exc:
    if config.get("fail_on_partial") or exc.steps_received == 0:
        raise
    logger.warning("Keeping partial forecast (%d steps): %s", exc.steps_received, exc)
