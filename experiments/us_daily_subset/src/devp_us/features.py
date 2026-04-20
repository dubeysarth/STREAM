from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERA5_CORE = [
    "dynamic_ERA5_total_precipitation",
    "dynamic_ERA5_potential_evaporation",
    "dynamic_ERA5_2m_temperature",
    "dynamic_ERA5_2m_dewpoint_temperature",
    "dynamic_ERA5_surface_pressure",
    "dynamic_ERA5_10m_u_component_of_wind",
    "dynamic_ERA5_10m_v_component_of_wind",
    "dynamic_ERA5_surface_net_solar_radiation",
    "dynamic_ERA5_surface_net_thermal_radiation",
]

ERA5_WATER_BALANCE = [
    "dynamic_ERA5_runoff",
    "dynamic_ERA5_surface_runoff",
    "dynamic_ERA5_sub_surface_runoff",
    "dynamic_ERA5_evaporation",
    "dynamic_ERA5_volumetric_soil_water_layer_1",
    "dynamic_ERA5_volumetric_soil_water_layer_2",
    "dynamic_ERA5_volumetric_soil_water_layer_3",
    "dynamic_ERA5_volumetric_soil_water_layer_4",
]

ERA5_SNOW = [
    "dynamic_ERA5_snow_depth",
    "dynamic_ERA5_snowfall",
    "dynamic_ERA5_snowmelt",
]

TIME_ENCODING_FEATURES = [
    "encoding_solar_insolation",
    "encoding_sine_month",
    "encoding_sine_dayofyear",
    "encoding_sine_weekofyear",
]

MONTHLY_TIME_ENCODING_FEATURES = [
    "encoding_solar_insolation",
    "encoding_sine_month",
]

WATER_BALANCE_FEATURES = {
    "precip": "dynamic_ERA5_total_precipitation",
    "pet": "dynamic_ERA5_potential_evaporation",
    "swi": "dynamic_GloFAS_soil_wetness_index",
}

HUMAN_USE_FEATURES = [
    "human_fracirrigated",
    "human_fracsealed",
    "human_fracwater",
    "human_reservoir_mask",
    "human_lake_mask",
    "human_fracgwused",
    "human_fracncused",
    "human_dom_mean",
    "human_dom_std",
    "human_ene_mean",
    "human_ene_std",
    "human_ind_mean",
    "human_ind_std",
    "human_liv_mean",
    "human_liv_std",
]


@dataclass(frozen=True)
class FeatureRegistry:
    """Named feature groups used across data builders and training scripts."""

    dynamic_groups: dict[str, list[str]]
    static_groups: dict[str, list[str]]
    time_encoding_features: list[str]

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "FeatureRegistry":
        return cls(
            dynamic_groups={k: list(v) for k, v in payload["dynamic_groups"].items()},
            static_groups={k: list(v) for k, v in payload["static_groups"].items()},
            time_encoding_features=list(payload.get("time_encoding_features", TIME_ENCODING_FEATURES)),
        )

    def dynamic(self, name: str) -> list[str]:
        return list(self.dynamic_groups[name])

    def time_features_for_frequency(self, frequency: str = "daily") -> list[str]:
        if frequency == "monthly":
            return list(MONTHLY_TIME_ENCODING_FEATURES)
        return list(self.time_encoding_features)

    def dynamic_for_frequency(self, name: str, frequency: str = "daily") -> list[str]:
        return list(self.dynamic_groups[name])

    def dynamic_with_time(self, name: str, frequency: str = "daily") -> list[str]:
        out = self.dynamic_for_frequency(name, frequency=frequency)
        for feature in self.time_features_for_frequency(frequency=frequency):
            if feature not in out:
                out.append(feature)
        return out

    def static(self, name: str) -> list[str]:
        return list(self.static_groups[name])

    def all_dynamic_union(self) -> list[str]:
        out: list[str] = []
        for features in self.dynamic_groups.values():
            for feature in features:
                if feature not in out:
                    out.append(feature)
        return out

    def all_dynamic_union_with_time(self) -> list[str]:
        out = self.all_dynamic_union()
        for feature in self.time_encoding_features:
            if feature not in out:
                out.append(feature)
        return out

    def resolve_static_names(self, group_name: str, available_names: list[str]) -> list[str]:
        base_names = [name for name in available_names if name not in HUMAN_USE_FEATURES]
        requested = self.static_groups.get(group_name, [])
        if not requested:
            return base_names
        resolved = list(base_names)
        for name in requested:
            if name in available_names and name not in resolved:
                resolved.append(name)
        return resolved
