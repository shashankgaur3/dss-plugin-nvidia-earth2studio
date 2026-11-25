# -*- coding: utf-8 -*-
"""Fetch initial conditions (ERA5 / GFS) for the preset's model into a managed folder."""
import logging

from dataiku.customrecipe import get_recipe_config

from earth2_dku.dku import params
from earth2_dku.sources.cache import fetch_initial_condition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("earth2.fetch")

config = get_recipe_config()
conn = params.connection_from_config(config)
model = params.build_model(conn)
source = params.build_source(config)
storage = params.folder_storage("initial_conditions", output=True)

for init_time in params.resolve_init_times(config, source):
    ref, cached = fetch_initial_condition(source, storage, model, init_time)
    logger.info("%s %s", "Reused" if cached else "Fetched", ref.path)
