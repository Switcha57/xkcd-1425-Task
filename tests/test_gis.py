import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from birdcheck.gis import (
    check_gis,
    extract_gps,
    find_park,
    load_parks,
    point_in_polygon,
    point_in_ring,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
HOLE = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]


def image_with_gps(path, lat_dms, lat_ref, lon_dms, lon_ref):
    image = Image.new("RGB", (32, 32), (120, 60, 30))
    exif = Image.Exif()
    exif[0x8825] = {
        1: lat_ref,
        2: tuple(IFDRational(n, d) for n, d in lat_dms),
        3: lon_ref,
        4: tuple(IFDRational(n, d) for n, d in lon_dms),
        6: 0,
    }
    image.save(path, exif=exif.tobytes())


class TestPointInPolygon(unittest.TestCase):
    def test_point_inside_square(self):
        self.assertTrue(point_in_ring(5, 5, SQUARE))

    def test_point_outside_square(self):
        self.assertFalse(point_in_ring(15, 5, SQUARE))

    def test_polygon_with_hole(self):
        rings = [SQUARE, HOLE]
        self.assertTrue(point_in_polygon(2, 2, rings))
        self.assertFalse(point_in_polygon(5, 5, rings))


class TestExtractGps(unittest.TestCase):
    def test_no_exif_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.jpg"
            Image.new("RGB", (16, 16)).save(path)
            self.assertIsNone(extract_gps(path))

    def test_gps_dms_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps.jpg"
            image_with_gps(
                path,
                lat_dms=[(44, 1), (27, 1), (378, 10)],
                lat_ref="N",
                lon_dms=[(110, 1), (49, 1), (4188, 100)],
                lon_ref="W",
            )
            lat, lon = extract_gps(path)
            self.assertAlmostEqual(lat, 44.4605, places=4)
            self.assertAlmostEqual(lon, -110.8283, places=4)

    def test_south_and_east_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps2.jpg"
            image_with_gps(
                path,
                lat_dms=[(33, 1), (52, 1), (7, 1)],
                lat_ref="S",
                lon_dms=[(151, 1), (12, 1), (30, 1)],
                lon_ref="E",
            )
            lat, lon = extract_gps(path)
            self.assertAlmostEqual(lat, -(33 + 52 / 60 + 7 / 3600), places=6)
            self.assertAlmostEqual(lon, 151 + 12 / 60 + 30 / 3600, places=6)


class TestParksData(unittest.TestCase):
    def test_bundled_parks_load(self):
        parks = load_parks()
        self.assertGreaterEqual(len(parks), 60)
        names = {name for name, _ in parks}
        self.assertIn("Yellowstone National Park", names)
        self.assertIn("Yosemite National Park", names)

    def test_find_park_known_points(self):
        self.assertEqual(
            find_park(44.4605, -110.8283), "Yellowstone National Park"
        )
        self.assertEqual(find_park(36.0544, -112.1401), "Grand Canyon National Park")
        self.assertIsNone(find_park(40.7128, -74.0060))
        self.assertIsNone(find_park(41.9028, 12.4964))


class TestCheckGis(unittest.TestCase):
    def test_no_gps_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.jpg"
            Image.new("RGB", (16, 16)).save(path)
            result = check_gis(path)
            self.assertIsNone(result.lat)
            self.assertIsNone(result.lon)
            self.assertIsNone(result.park)
            self.assertIsNone(result.in_national_park)

    def test_gps_inside_park(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps.jpg"
            image_with_gps(
                path,
                lat_dms=[(44, 1), (27, 1), (378, 10)],
                lat_ref="N",
                lon_dms=[(110, 1), (49, 1), (4188, 100)],
                lon_ref="W",
            )
            result = check_gis(path)
            self.assertAlmostEqual(result.lat, 44.4605, places=4)
            self.assertAlmostEqual(result.lon, -110.8283, places=4)
            self.assertEqual(result.park, "Yellowstone National Park")
            self.assertTrue(result.in_national_park)


if __name__ == "__main__":
    unittest.main()
