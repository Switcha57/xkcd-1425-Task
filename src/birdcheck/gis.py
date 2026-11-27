import json
from dataclasses import dataclass
from importlib import resources

from PIL import Image

GPS_IFD_TAG = 0x8825


@dataclass
class GisResult:
    path: str
    lat: float | None
    lon: float | None
    park: str | None
    in_national_park: bool | None


def _dms_to_decimal(value, ref):
    degrees, minutes, seconds = (float(v) for v in value)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(path):
    with Image.open(path) as image:
        gps = image.getexif().get_ifd(GPS_IFD_TAG)
    if not gps:
        return None
    lat = gps.get(2)
    lon = gps.get(4)
    if lat is None or lon is None:
        return None
    lat_ref = gps.get(1, "N")
    lon_ref = gps.get(3, "E")
    return _dms_to_decimal(lat, lat_ref), _dms_to_decimal(lon, lon_ref)


def load_parks():
    text = resources.files("birdcheck").joinpath("national_parks.geojson").read_text()
    collection = json.loads(text)
    parks = []
    for feature in collection["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        else:
            polygons = geometry["coordinates"]
        parks.append((feature["properties"]["name"], polygons))
    return parks


def point_in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_int = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_int:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lon, lat, rings):
    if not point_in_ring(lon, lat, rings[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in rings[1:])


def find_park(lat, lon, parks=None):
    if parks is None:
        parks = load_parks()
    for name, polygons in parks:
        if any(point_in_polygon(lon, lat, rings) for rings in polygons):
            return name
    return None


def check_gis(path, parks=None):
    gps = extract_gps(path)
    if gps is None:
        return GisResult(path=str(path), lat=None, lon=None, park=None, in_national_park=None)
    lat, lon = gps
    park = find_park(lat, lon, parks)
    return GisResult(
        path=str(path),
        lat=lat,
        lon=lon,
        park=park,
        in_national_park=park is not None,
    )
