import unittest
import unittest.mock

from opt_utils import gen_meta_prompt


class TestGenMetaPrompt(unittest.TestCase):
    def setUp(self):
        self.domain = "joke"
        self.old_instructions_and_scores = [
            ("instruction1", 25, -1),
            ("instruction2", 0, -1),
            ("instruction3", 75, -1),
        ]
        self.old_instruction_score_threshold = 0.3
        self.max_num_instructions = 2

    def test_gen_meta_prompt(self):
        expected_prompt = (
            "meta_prompt_content\n\n"
            "text:instruction1\n"
            "score:25\n"
            "text:instruction3\n"
            "score:75\n\n"
            "remaining_meta_prompt_content"
        )

        with unittest.mock.patch(
            "builtins.open",
            unittest.mock.mock_open(
                read_data="meta_prompt_content\nremaining_meta_prompt_content"
            ),
        ):
            result = gen_meta_prompt(
                domain=self.domain,
                old_instructions_and_scores=self.old_instructions_and_scores,
                old_instruction_score_threshold=self.old_instruction_score_threshold,
                max_num_instructions=self.max_num_instructions,
            )
            self.assertEqual(result, expected_prompt)


if __name__ == "__main__":
    unittest.main()
