# -*- coding: utf-8 -*-
"""Macro: run all preflight checks for a NIM connection preset and return an HTML table."""
from dataiku.runnables import Runnable

from earth2_dku.dku import params
from earth2_dku.errors import Earth2Error
from earth2_dku.models.registry import get_model
from earth2_dku.validation.preflight import render_html, run_preflight


class MyRunnable(Runnable):
    def __init__(self, project_key, config, plugin_config):
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        conn = params.connection_from_config(self.config)
        try:
            model = get_model(conn.model_id)
        except Earth2Error:
            model = None
        try:
            client = params.build_client(conn, model)
        except Earth2Error:
            client = None
        return render_html(run_preflight(
            conn.model_id, client, conn.gpu_memory_gb,
            params.build_source(self.config), self.config.get("init_time"),
        ))
