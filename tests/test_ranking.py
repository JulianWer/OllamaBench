import unittest
import math
from collections import defaultdict
from evaluation.ranking import calculate_mELO_ratings, C_ELO_CONSTANT

class TestRanking(unittest.TestCase):


    def test_calculate_mELO_ratings_basic(self):
        """
        Testet die m-ELO-Berechnung mit einem einfachen Satz von Matches.
        """
        model_ratings = {}
        matches = [
            ("model_a", "model_b", "coding", 1.0),
            ("model_a", "model_b", "coding", 1.0),
            ("model_b", "model_a", "coding", 0.0),
        ]
        
        initial_rating = 1000.0
        
        calculate_mELO_ratings(
            model_ratings=model_ratings,
            all_matches=matches,
            initial_rating=initial_rating,
            learning_rate=50,
            epochs=100
        )

        rating_a = model_ratings["model_a"]["elo_rating_by_category"]["coding"]
        rating_b = model_ratings["model_b"]["elo_rating_by_category"]["coding"]

        self.assertGreater(rating_a, initial_rating)
        self.assertLess(rating_b, initial_rating)
        
        self.assertEqual(model_ratings["model_a"]["wins"], 3)
        self.assertEqual(model_ratings["model_a"]["losses"], 0)
        self.assertEqual(model_ratings["model_a"]["num_comparisons"], 3)
        
        self.assertEqual(model_ratings["model_b"]["wins"], 0)
        self.assertEqual(model_ratings["model_b"]["losses"], 3)
        self.assertEqual(model_ratings["model_b"]["num_comparisons"], 3)


    def test_calculate_mELO_with_draws_and_categories(self):
        """
        Testet die m-ELO-Berechnung mit Unentschieden und mehreren Kategorien.
        """
        model_ratings = {}
        matches = [
            ("model_a", "model_b", "coding", 1.0),
            ("model_a", "model_b", "writing", 0.5),
            ("model_c", "model_a", "coding", 0.0),
            ("model_c", "model_b", "writing", 1.0),
        ]
        
        initial_rating = 1000.0
        
        calculate_mELO_ratings(
            model_ratings=model_ratings,
            all_matches=matches,
            initial_rating=initial_rating,
            learning_rate=50,
            epochs=100
        )
        
        rating_a_coding = model_ratings["model_a"]["elo_rating_by_category"]["coding"]
        rating_b_coding = model_ratings["model_b"]["elo_rating_by_category"]["coding"]
        rating_c_coding = model_ratings["model_c"]["elo_rating_by_category"]["coding"]
        self.assertGreater(rating_a_coding, rating_b_coding)
        self.assertGreater(rating_a_coding, rating_c_coding)

        rating_a_writing = model_ratings["model_a"]["elo_rating_by_category"]["writing"]
        rating_b_writing = model_ratings["model_b"]["elo_rating_by_category"]["writing"]
        rating_c_writing = model_ratings["model_c"]["elo_rating_by_category"]["writing"]
        self.assertGreater(rating_c_writing, rating_b_writing)
        self.assertGreater(rating_c_writing, rating_a_writing)

        self.assertEqual(model_ratings["model_a"]["wins"], 2)
        self.assertEqual(model_ratings["model_a"]["losses"], 0)
        self.assertEqual(model_ratings["model_a"]["draws"], 1)
        self.assertEqual(model_ratings["model_a"]["num_comparisons"], 3)

        self.assertEqual(model_ratings["model_c"]["wins"], 1)
        self.assertEqual(model_ratings["model_c"]["losses"], 1)
        self.assertEqual(model_ratings["model_c"]["draws"], 0)
        self.assertEqual(model_ratings["model_c"]["num_comparisons"], 2)
        
        self.assertEqual(model_ratings["model_b"]["wins"], 0)
        self.assertEqual(model_ratings["model_b"]["losses"], 2)
        self.assertEqual(model_ratings["model_b"]["draws"], 1)
        self.assertEqual(model_ratings["model_b"]["num_comparisons"], 3)

if __name__ == '__main__':
    unittest.main()