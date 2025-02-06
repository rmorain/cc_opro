import unittest
import unittest.mock
from unittest.mock import patch

import pandas as pd

from eval_utils import evaluate_single_instruction, is_joke_online


class TestEvaluateSingleInstruction(unittest.TestCase):
    def setUp(self):
        self.instruction = "Tell me a joke."
        self.domain = "joke"
        self.n_artifacts = 1

    def mock_call_server_func(self, prompt):
        if "<INS>" in prompt:
            return "Strongly Agree"
        return "This is a joke."

    @patch("eval_utils.os.listdir", return_value=["funny.eval"])
    @patch("eval_utils.open", unittest.mock.mock_open(read_data="<INS>"))
    @patch("os.listdir")  # Add this decorator if you're mocking os.listdir
    def test_evaluate_single_instruction(self, mock_open, mock_listdir):
        result_df = evaluate_single_instruction(
            instruction=self.instruction,
            call_server_func=self.mock_call_server_func,
            n_artifacts=self.n_artifacts,
            domain=self.domain,
        )
        expected_data = {
            "funny": [100],
            "artifact": ["This is a joke."],
            "domain": ["joke"],
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result_df, expected_df)

    @patch("eval_utils.search")
    def test_is_joke_online(self, mock_search):
        joke_text = "Why did the chicken cross the road?"
        mock_search.return_value = iter(
            ["http://example.com/joke1", "http://example.com/joke2"]
        )

        result = is_joke_online(joke_text)
        self.assertTrue(result)

        mock_search.return_value = iter([])
        result = is_joke_online(joke_text)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
