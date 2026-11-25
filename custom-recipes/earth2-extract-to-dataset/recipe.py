# -*- coding: utf-8 -*-
"""Extract forecast values at sites or over a region into a dataset."""
import logging

from dataiku.customrecipe import get_recipe_config

from earth2_dku.dku import params
from earth2_dku.dku.datasets import write_dataset
from earth2_dku.extract.pipeline import extract_forecast, output_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("earth2.extract")

config = get_recipe_config()
storage = params.folder_storage("forecast")
options = params.extract_options(config)
sites = params.read_sites(config) if options.mode == "points" else None

df = extract_forecast(storage, options, sites)
write_dataset("forecast_values", df, output_schema(df))
logger.info("Wrote %d rows", len(df))
