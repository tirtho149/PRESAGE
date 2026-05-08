"""
utils/geo.py
============
Geospatial utilities for Pathome (paper §6.3, Layer 3).

Concerns:
1. Resolving the Bugwood CSV ``Location`` (US state name) to a
   representative (lat, lon) centroid — this is the geo source actually
   available in ``BugWood_Diseases.csv``. EXIF GPS extraction is retained
   for the legacy folder-tree path.
2. Mapping (lat, lon) → FAO agro-ecological zone (AEZ).
3. Climate-zone vector + similarity for Layer-5 geo-weighted retrieval
   (Eq. retrieval: 0.7 · cos + 0.3 · ClimSim).

The FAO AEZ shapefile is not bundled. Set ``PATHOME_AEZ_SHAPEFILE`` to a
local path of FAO_AEZv4_50K.shp (download from FAO GAEZ portal). Without
the shapefile, ``aez_lookup`` falls back to a coarse latitude band table.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EXIF GPS extraction
# ---------------------------------------------------------------------------

def _to_decimal(coord, ref) -> Optional[float]:
    """Convert EXIF DMS triple + N/S/E/W reference to signed decimal degrees."""
    if coord is None:
        return None
    try:
        d, m, s = (float(x) for x in coord)
    except Exception:
        return None
    val = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        val = -val
    return val


def extract_gps_from_image(path: str) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
    """
    Read EXIF GPS lat/lon and capture timestamp from a JPEG.
    Returns (lat, lon, dt) — any of which may be None when EXIF is missing.
    """
    try:
        from PIL import Image, ExifTags
    except ImportError as e:  # pragma: no cover
        raise ImportError("Pillow is required for EXIF parsing") from e

    try:
        img = Image.open(path)
    except Exception:
        return None, None, None

    exif = img._getexif() if hasattr(img, "_getexif") else None
    if not exif:
        return None, None, None

    tag_for = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    gps_raw = tag_for.get("GPSInfo")
    lat = lon = None
    if gps_raw:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        lat = _to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        lon = _to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))

    ts = tag_for.get("DateTimeOriginal") or tag_for.get("DateTime")
    dt = None
    if ts:
        try:
            dt = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            dt = None

    return lat, lon, dt


# ---------------------------------------------------------------------------
# AEZ lookup (FAO 0.5° grid)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AEZ:
    """FAO agro-ecological zone identifier."""
    code: str        # e.g. "TLM" tropical lowland moist
    region: str      # human-readable region name
    climate: str     # climate descriptor (humid_tropical, cool_humid, ...)


# A coarse fallback table used when the shapefile is unavailable. It is good
# enough for prototyping but the paper claim of EPPO r >= 0.71 requires the
# real shapefile.
_FALLBACK_AEZ = [
    # (lat_lo, lat_hi, lon_lo, lon_hi, AEZ)
    (-15.0,  15.0, -180.0, 180.0, AEZ("TLM",  "Tropics (lowland moist)",   "humid_tropical")),
    ( 15.0,  35.0, -180.0, 180.0, AEZ("STM",  "Subtropics (moist)",        "subtropical_moist")),
    (-35.0, -15.0, -180.0, 180.0, AEZ("STM",  "Subtropics (moist)",        "subtropical_moist")),
    ( 35.0,  55.0, -180.0, 180.0, AEZ("TMP",  "Temperate humid",           "temperate_humid")),
    (-55.0, -35.0, -180.0, 180.0, AEZ("TMP",  "Temperate humid",           "temperate_humid")),
    ( 55.0,  90.0, -180.0, 180.0, AEZ("BOR",  "Boreal / cool",             "cool_humid")),
    (-90.0, -55.0, -180.0, 180.0, AEZ("BOR",  "Boreal / cool",             "cool_humid")),
]


def aez_lookup(lat: float, lon: float) -> Optional[AEZ]:
    """
    Map (lat, lon) → FAO AEZ. Uses the FAO shapefile if PATHOME_AEZ_SHAPEFILE
    is set and geopandas is available; otherwise falls back to a coarse grid.
    """
    if lat is None or lon is None:
        return None

    shp = os.environ.get("PATHOME_AEZ_SHAPEFILE")
    if shp and os.path.exists(shp):
        try:
            import geopandas as gpd  # type: ignore
            from shapely.geometry import Point  # type: ignore
        except ImportError:
            shp = None
        else:
            # TODO(pathome): cache the GeoDataFrame across calls
            gdf = gpd.read_file(shp)
            pt = Point(lon, lat)
            hit = gdf[gdf.contains(pt)]
            if len(hit):
                row = hit.iloc[0]
                return AEZ(
                    code=str(row.get("AEZ_CODE", row.get("CODE", "UNK"))),
                    region=str(row.get("REGION", "")),
                    climate=str(row.get("CLIMATE", "")),
                )

    for la0, la1, lo0, lo1, aez in _FALLBACK_AEZ:
        if la0 <= lat < la1 and lo0 <= lon < lo1:
            return aez
    return None


# ---------------------------------------------------------------------------
# Climate vector for retrieval similarity (paper Eq. retrieval)
# ---------------------------------------------------------------------------

# Six-dimensional climate descriptor: (mean_temp_norm, temp_seasonality,
# annual_precip_norm, precip_seasonality, humidity, elevation_norm).
# These values are placeholders pending integration with WorldClim BIO vars.
_CLIMATE_VECTORS = {
    "humid_tropical":     np.array([0.85, 0.10, 0.90, 0.30, 0.85, 0.20]),
    "subtropical_moist":  np.array([0.65, 0.30, 0.65, 0.45, 0.65, 0.30]),
    "temperate_humid":    np.array([0.45, 0.55, 0.60, 0.40, 0.60, 0.35]),
    "cool_humid":         np.array([0.25, 0.65, 0.55, 0.35, 0.65, 0.40]),
    "cool_dry":           np.array([0.30, 0.70, 0.20, 0.30, 0.30, 0.50]),
    "tropical_lowland":   np.array([0.85, 0.10, 0.85, 0.25, 0.85, 0.15]),
}


def climate_vector(lat: float, lon: float) -> np.ndarray:
    """Return a 6-dim climate descriptor for use in Eq. retrieval ClimSim."""
    aez = aez_lookup(lat, lon)
    if aez is None or aez.climate not in _CLIMATE_VECTORS:
        return np.zeros(6, dtype=np.float32)
    return _CLIMATE_VECTORS[aez.climate].astype(np.float32)


def clim_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Climate-zone similarity in [0, 1]."""
    if v1.size == 0 or v2.size == 0:
        return 0.0
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(v1, v2) / (n1 * n2))))


# ---------------------------------------------------------------------------
# Encoding helpers for OBSERVE input phi_geo
# ---------------------------------------------------------------------------

def encode_phi_geo(lat: Optional[float], lon: Optional[float], month: Optional[int]) -> np.ndarray:
    """
    Build the phi_geo input passed to OBSERVE alongside image + context.
    Currently: [climate_vector_6d, sin(2pi*month/12), cos(2pi*month/12)].
    Returns an 8-dim float32 vector. Missing values → zeros.
    """
    out = np.zeros(8, dtype=np.float32)
    if lat is not None and lon is not None:
        out[:6] = climate_vector(float(lat), float(lon))
    if month is not None and 1 <= int(month) <= 12:
        m = int(month)
        out[6] = math.sin(2.0 * math.pi * m / 12.0)
        out[7] = math.cos(2.0 * math.pi * m / 12.0)
    return out


# ---------------------------------------------------------------------------
# US-state centroid lookup (Bugwood CSV "Location" column)
# ---------------------------------------------------------------------------
#
# Bugwood image records carry a US state name (e.g. "North Carolina") rather
# than per-photo GPS, so the Layer-3 epidemiology grid is at state resolution.
# Centroids are population-weighted geographic centers (US Census 2020) for
# the 50 states + DC, plus US territories that occasionally appear in the
# Bugwood feed. Used by data/bugwood_loader.py and downstream phi_geo /
# aez_lookup calls when no finer GPS is available.

US_STATE_CENTROID: dict = {
    "alabama":              (32.806671,  -86.791130),
    "alaska":               (61.370716, -152.404419),
    "arizona":              (33.729759, -111.431221),
    "arkansas":             (34.969704,  -92.373123),
    "california":           (36.116203, -119.681564),
    "colorado":             (39.059811, -105.311104),
    "connecticut":          (41.597782,  -72.755371),
    "delaware":             (39.318523,  -75.507141),
    "district of columbia": (38.897438,  -77.026817),
    "florida":              (27.766279,  -81.686783),
    "georgia":              (33.040619,  -83.643074),
    "hawaii":               (21.094318, -157.498337),
    "idaho":                (44.240459, -114.478828),
    "illinois":             (40.349457,  -88.986137),
    "indiana":              (39.849426,  -86.258278),
    "iowa":                 (42.011539,  -93.210526),
    "kansas":               (38.526600,  -96.726486),
    "kentucky":             (37.668140,  -84.670067),
    "louisiana":            (31.169546,  -91.867805),
    "maine":                (44.693947,  -69.381927),
    "maryland":             (39.063946,  -76.802101),
    "massachusetts":        (42.230171,  -71.530106),
    "michigan":             (43.326618,  -84.536095),
    "minnesota":            (45.694454,  -93.900192),
    "mississippi":          (32.741646,  -89.678696),
    "missouri":             (38.456085,  -92.288368),
    "montana":              (46.921925, -110.454353),
    "nebraska":             (41.125370,  -98.268082),
    "nevada":               (38.313515, -117.055374),
    "new hampshire":        (43.452492,  -71.563896),
    "new jersey":           (40.298904,  -74.521011),
    "new mexico":           (34.840515, -106.248482),
    "new york":             (42.165726,  -74.948051),
    "north carolina":       (35.630066,  -79.806419),
    "north dakota":         (47.528912, -99.784012),
    "ohio":                 (40.388783,  -82.764915),
    "oklahoma":             (35.565342,  -96.928917),
    "oregon":               (44.572021, -122.070938),
    "pennsylvania":         (40.590752,  -77.209755),
    "rhode island":         (41.680893,  -71.511780),
    "south carolina":       (33.856892,  -80.945007),
    "south dakota":         (44.299782,  -99.438828),
    "tennessee":            (35.747845,  -86.692345),
    "texas":                (31.054487,  -97.563461),
    "utah":                 (40.150032, -111.862434),
    "vermont":              (44.045876,  -72.710686),
    "virginia":             (37.769337,  -78.169968),
    "washington":           (47.400902, -121.490494),
    "west virginia":        (38.491226,  -80.954453),
    "wisconsin":            (44.268543,  -89.616508),
    "wyoming":              (42.755966, -107.302490),
    "puerto rico":          (18.220833,  -66.590149),
    "guam":                 (13.444304, 144.793732),
    "us virgin islands":    (18.335765,  -64.896335),
}


def state_to_latlon(state: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Resolve a US state name to a (lat, lon) centroid.

    Accepts free-form casing / whitespace. Returns ``(None, None)`` when the
    state is unknown (foreign locations, blanks). The centroid is the state's
    population-weighted center, which is sufficient for AEZ lookup at this
    layer-3 grid resolution.
    """
    if not state:
        return None, None
    key = str(state).strip().lower()
    if not key:
        return None, None
    hit = US_STATE_CENTROID.get(key)
    if hit is not None:
        return hit
    # Tolerate "St." / "Saint" / hyphenation variants
    cleaned = key.replace(".", "").replace("st ", "saint ")
    return US_STATE_CENTROID.get(cleaned, (None, None))
