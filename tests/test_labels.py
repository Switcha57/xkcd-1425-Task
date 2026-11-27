import unittest

from birdcheck.labels import BIRD_CLASS_IDS, load_labels


class TestLabels(unittest.TestCase):
    def test_labels_count(self):
        labels = load_labels()
        self.assertEqual(len(labels), 1000)
        self.assertEqual(labels[0], "tench")
        self.assertEqual(labels[15], "robin")
        self.assertEqual(labels[999], "toilet tissue")

    def test_bird_class_ids(self):
        for bird in (7, 8, 9, 15, 22, 84, 94, 99, 130, 145, 146):
            self.assertIn(bird, BIRD_CLASS_IDS)
        for not_bird in (0, 2, 147, 258, 600, 999):
            self.assertNotIn(not_bird, BIRD_CLASS_IDS)


if __name__ == "__main__":
    unittest.main()
