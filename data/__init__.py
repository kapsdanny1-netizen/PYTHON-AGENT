"""Synthetic data package: demo fleet registry, generators, anomaly injection."""

from data.generators import (
    CHANNELS,
    FLEET,
    SCENARIO_WT07_BEARING,
    AnomalyKind,
    AnomalySpec,
    AssetSpec,
    AssetType,
    GenerateRequest,
    GenerationSummary,
    dataframe_to_sensor_rows,
    generate_and_store,
    generate_sensor_data,
    get_asset,
    store_sensor_dataframe,
)

__all__ = [
    "CHANNELS",
    "FLEET",
    "SCENARIO_WT07_BEARING",
    "AnomalyKind",
    "AnomalySpec",
    "AssetSpec",
    "AssetType",
    "GenerateRequest",
    "GenerationSummary",
    "dataframe_to_sensor_rows",
    "generate_and_store",
    "generate_sensor_data",
    "get_asset",
    "store_sensor_dataframe",
]
