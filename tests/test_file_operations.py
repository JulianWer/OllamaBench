import unittest
from unittest.mock import patch
from utils.file_operations import _deep_merge_dicts, save_elo_results

class TestFileOperations(unittest.TestCase):
    """
    Test-Suite für die Dateioperationen. 
    Diese Tests verwenden jetzt einen robusteren Mocking-Ansatz.
    """

    def test_deep_merge_dicts(self):
        """
        Testet die rekursive Zusammenführung von Dictionaries.
        Dieser Test war bereits korrekt und bleibt unverändert.
        """
        target = {
            "a": 1,
            "b": {"x": 10, "y": 20},
            "d": [1, 2]
        }
        source = {
            "b": {"y": 25, "z": 30},
            "c": 3,
            "d": [3, 4]
        }
        
        _deep_merge_dicts(target, source)
        
        expected = {
            "a": 1,
            "b": {"x": 10, "y": 25, "z": 30},
            "c": 3,
            "d": [3, 4] # Liste wird überschrieben, nicht zusammengeführt
        }
        
        self.assertEqual(target, expected)

    @patch('utils.file_operations._write_json_internal')
    @patch('utils.file_operations._read_json_internal')
    @patch('utils.file_operations.filelock.FileLock')
    def test_save_elo_results_new_file(self, mock_filelock, mock_read, mock_write):
        """
        Testet das Speichern von ELO-Ergebnissen, wenn die Zieldatei nicht existiert.
        Dieser Test ist neu geschrieben, um robuster zu sein.
        """
        mock_read.return_value = None
        mock_write.return_value = True

        config = {
            "paths": {
                "results_file": "fake/results.json",
                "lock_file": "fake/results.lock"
            }
        }
        new_ratings = {
            "model_a": {"elo_rating_by_category": {"coding": 1010}}
        }

        success = save_elo_results(new_ratings, config)
        self.assertTrue(success)

        # Check locking
        mock_filelock.return_value.acquire.assert_called_once()
        mock_filelock.return_value.release.assert_called_once()
        
        # Check reading file
        mock_read.assert_called_once_with("fake/results.json")
        
        # Check Data for writing function
        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][0] 
        file_path = mock_write.call_args[0][1]   

        self.assertEqual(file_path, "fake/results.json")
        self.assertIn("models", written_data)
        self.assertIn("timestamp", written_data)
        self.assertEqual(written_data["models"]["model_a"]["elo_rating_by_category"]["coding"], 1010)

    @patch('utils.file_operations._write_json_internal')
    @patch('utils.file_operations._read_json_internal')
    @patch('utils.file_operations.filelock.FileLock')
    def test_save_elo_results_merge_file(self, mock_filelock, mock_read, mock_write):
        """
        Testet das korrekte Zusammenführen von neuen Daten mit existierenden ELO-Ergebnissen.
        Dieser Testfall ist neu und verbessert die Testabdeckung.
        """
        # Simulate, reading file
        existing_data = {
            "models": {
                "model_b": {"elo_rating_by_category": {"writing": 950}},
                "model_a": {"elo_rating_by_category": {"coding": 1000}} # old value
            },
            "timestamp": "old_timestamp"
        }
        mock_read.return_value = existing_data
        mock_write.return_value = True

        config = {
            "paths": { "results_file": "fake/results.json", "lock_file": "fake/results.lock" }
        }

        new_ratings = {
            "model_a": {"elo_rating_by_category": {"coding": 1015, "writing": 1005}},
            "model_c": {"elo_rating_by_category": {"coding": 1100}}
        }

        success = save_elo_results(new_ratings, config)
        self.assertTrue(success)

        mock_read.assert_called_once_with("fake/results.json")
        mock_write.assert_called_once()
        
        written_data = mock_write.call_args[0][0]

        final_models = written_data["models"]
        self.assertEqual(final_models["model_b"]["elo_rating_by_category"]["writing"], 950) 
        self.assertEqual(final_models["model_a"]["elo_rating_by_category"]["coding"], 1015) 
        self.assertEqual(final_models["model_a"]["elo_rating_by_category"]["writing"], 1005) 
        self.assertEqual(final_models["model_c"]["elo_rating_by_category"]["coding"], 1100) 
        self.assertNotEqual(written_data["timestamp"], "old_timestamp") 

if __name__ == '__main__':
    unittest.main()
