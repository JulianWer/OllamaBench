import unittest
from utils.get_data import prepare_prompt_for_llm, extract_prompt_text

class TestGetData(unittest.TestCase):

    def test_prepare_prompt_for_llm(self):
        """
        Testet die Konvertierung von Roh-Prompts in das für den LLM erwartete Format.
        """
        self.assertEqual(prepare_prompt_for_llm("Hallo Welt"), "Hallo Welt")

        multi_turn_input = ["Frage 1", "Antwort 1", "Frage 2"]
        expected_output = [
            {"role": "user", "content": "Frage 1"},
            {"role": "assistant", "content": "Antwort 1"},
            {"role": "user", "content": "Frage 2"}
        ]
        self.assertEqual(prepare_prompt_for_llm(multi_turn_input), expected_output)

        formatted_input = [{"role": "user", "content": "Test"}]
        self.assertEqual(prepare_prompt_for_llm(formatted_input), formatted_input)
        
        self.assertIsNone(prepare_prompt_for_llm(123))
        self.assertIsNone(prepare_prompt_for_llm({"invalid": "format"}))

    def test_extract_prompt_text(self):
        """
        Testet die Extraktion eines einzelnen Prompt-Textes für den Judge.
        """
        self.assertEqual(extract_prompt_text("Einfacher Prompt"), "Einfacher Prompt")

        self.assertEqual(extract_prompt_text(["Zeile 1", "Zeile 2"]), "Zeile 1\nZeile 2")
        
        dict_prompt_user_last = [
            {"role": "user", "content": "Erste Frage"},
            {"role": "assistant", "content": "Erste Antwort"},
            {"role": "user", "content": "Zweite Frage"}
        ]
        self.assertEqual(extract_prompt_text(dict_prompt_user_last), "Zweite Frage")

        dict_prompt_no_user = [
            {"role": "system", "content": "System-Nachricht"},
            {"role": "assistant", "content": "Antwort"}
        ]
        self.assertEqual(extract_prompt_text(dict_prompt_no_user), "System-Nachricht\nAntwort")
        
        self.assertIsNone(extract_prompt_text([]))
        self.assertIsNone(extract_prompt_text([{"role": "user"}])) 
        self.assertIsNone(extract_prompt_text(None))

if __name__ == '__main__':
    unittest.main()