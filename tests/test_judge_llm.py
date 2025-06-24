import unittest
from models.judge_llm import _build_judge_prompt, JudgeLLM

class TestJudgeLLM(unittest.TestCase):

    def setUp(self):
        """Erstellt eine Instanz von JudgeLLM für Tests."""
        self.judge = JudgeLLM(api_url="http://fake-url:11434", model_name="test-judge")

    def test_build_judge_prompt_no_ground_truth(self):
        """
        Testet die Erstellung des Judge-Prompts ohne eine Referenzantwort.
        """
        prompt = "Was ist die Hauptstadt von Frankreich?"
        response1 = "Paris"
        response2 = "Die Hauptstadt ist Paris."
        
        judge_prompt = _build_judge_prompt(prompt, response1, response2)
        
        self.assertIn("[User Question]", judge_prompt)
        self.assertIn(prompt, judge_prompt)
        self.assertIn("[The Start of Assistant A’s Answer]", judge_prompt)
        self.assertIn(response1, judge_prompt)
        self.assertIn("[The Start of Assistant B’s Answer]", judge_prompt)
        self.assertIn(response2, judge_prompt)
        self.assertNotIn("[The Start of Reference Answer]", judge_prompt)
        self.assertIn("strictly following this format: '[[A]]' if assistant A is better, '[[B]]' if assistant B is better, and '[[C]]' for a tie.", judge_prompt)

    def test_build_judge_prompt_with_ground_truth(self):
        """
        Testet die Erstellung des Judge-Prompts MIT einer Referenzantwort.
        """
        prompt = "Fasse zusammen."
        response1 = "Kurze Zusammenfassung."
        response2 = "Lange Zusammenfassung."
        ground_truth = "Eine perfekte Zusammenfassung."
        
        judge_prompt = _build_judge_prompt(prompt, response1, response2, ground_truth=ground_truth)
        
        self.assertIn("[The Start of Reference Answer]", judge_prompt)
        self.assertIn(ground_truth, judge_prompt)
        self.assertIn("evaluate the correctness of the responses", judge_prompt)
        self.assertIn("'[[A]]' if assistant A is more correct, '[[B]]' if assistant B is more correct", judge_prompt)

    def test_parse_judge_verdict(self):
        """
        Testet das Parsen von verschiedenen Formaten der Judge-Antwort.
        """
        self.assertEqual(self.judge._parse_judge_verdict("[[A]]"), 1.0)
        self.assertEqual(self.judge._parse_judge_verdict("[[B]]"), 0.0)
        self.assertEqual(self.judge._parse_judge_verdict("[[C]]"), 0.5)

        self.assertEqual(self.judge._parse_judge_verdict("Ich denke, A ist besser. [[A]]"), 1.0)
        self.assertEqual(self.judge._parse_judge_verdict("Antwort B ist klar überlegen. [[B]]"), 0.0)
        self.assertEqual(self.judge._parse_judge_verdict("Beide sind gleich gut. [[C]] Dies war eine schwere Entscheidung."), 0.5)

        self.assertEqual(self.judge._parse_judge_verdict("  [[a]]  "), 1.0)
        self.assertEqual(self.judge._parse_judge_verdict("[[ b ]]"), 0.0)

        self.assertIsNone(self.judge._parse_judge_verdict("A ist besser."))
        self.assertIsNone(self.judge._parse_judge_verdict("[A]"))
        self.assertIsNone(self.judge._parse_judge_verdict(""))
        self.assertIsNone(self.judge._parse_judge_verdict(None))
        self.assertIsNone(self.judge._parse_judge_verdict("Das ist [[D]]."))

if __name__ == '__main__':
    unittest.main()