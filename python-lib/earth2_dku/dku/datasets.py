"""Writing DSS datasets."""
from __future__ import annotations

import pandas as pd

from ..errors import ConfigError


def write_dataset(role: str, df: pd.DataFrame, schema: list[dict]) -> None:
    """Write ``df`` to the dataset bound to ``role`` with an explicit schema."""
    import dataiku
    from dataiku.customrecipe import get_output_names_for_role

    names = get_output_names_for_role(role)
    if not names:
        raise ConfigError(f"Connect an output dataset to the '{role}' role.")
    df = df.copy()
    for col in df.columns:  # DSS stores dates as UTC; hand it naive UTC timestamps
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    dataset = dataiku.Dataset(names[0])
    dataset.write_schema(schema)
    with dataset.get_writer() as writer:
        writer.write_dataframe(df)
