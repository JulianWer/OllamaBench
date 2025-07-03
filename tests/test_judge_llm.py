import unittest
from unittest.mock import patch
from models.judge_llm import JudgeLLM

class TestJudgeLLMImproved(unittest.TestCase):

    def setUp(self):
        self.chat_patcher = patch('models.judge_llm.chat_with_model')
        self.mock_chat_with_model = self.chat_patcher.start()

        self.judge = JudgeLLM(
            api_url="http://fake-api.com/v1",
            model_name="test-judge-model",
            system_prompt="You are an impartial judge.",
            temperature=0.0,
            has_reasoning=False
        )

    def tearDown(self):
        self.chat_patcher.stop()

    def test_evaluate_outcomes(self):
        # Test cases: (llm_response_content, expected_score)
        test_cases = [
            ("[[A]]", 1.0),
            ("[[B]]", 0.0),
            ("[[C]]", 0.5),
            ("  [[A]]  ", 1.0),  # Test with extra whitespace
            ("Some noise before [[B]] and after.", 0.0), # Test with surrounding text
            ("The verdict is [[c]]", 0.5), # Test with case-insensitivity
            ("This is a malformed response without a verdict", None), # Test parsing failure
            ("", None), # Test empty response
        ]

        for response_text, expected_score in test_cases:
            with self.subTest(response_text=response_text, expected_score=expected_score):
                self.mock_chat_with_model.return_value = {
                    'message': {'content': response_text}
                }

                # Call the method under test
                score = self.judge.evaluate(
                    user_prompt="Is the sky blue?",
                    response_a="Yes, it is.",
                    response_b="No, it's green."
                )

                # Assert that the returned score is correct
                self.assertEqual(score, expected_score)
                self.mock_chat_with_model.assert_called_once()
                self.mock_chat_with_model.reset_mock()

    def test_evaluate_with_ground_truth(self):
        self.mock_chat_with_model.return_value = {'message': {'content': '[[A]]'}}
        ground_truth_text = "The sky is blue due to Rayleigh scattering."

        self.judge.evaluate(
            user_prompt="Is the sky blue?",
            response_a="Yes.",
            response_b="No.",
            ground_truth=ground_truth_text
        )

        self.mock_chat_with_model.assert_called_once()
        _args, kwargs = self.mock_chat_with_model.call_args
        self.assertIn(ground_truth_text, kwargs['prompt'])
        self.assertIn("[The Start of Reference Answer]", kwargs['prompt'])

    def test_evaluate_handles_api_failures(self):
        # Test cases for various invalid API responses
        failure_cases = [
            None,
            "just a string, not a dict",
            {},
            {"message": "not a dict"},
            {"message": {}}, # Missing 'content' key
        ]
        
        for api_response in failure_cases:
             with self.subTest(api_response=api_response):
                self.mock_chat_with_model.return_value = api_response
                
                score = self.judge.evaluate(
                    user_prompt="test prompt",
                    response_a="answer A",
                    response_b="answer B"
                )
                
                self.assertIsNone(score)
                self.mock_chat_with_model.reset_mock()

if __name__ == '__main__':
    unittest.main()
