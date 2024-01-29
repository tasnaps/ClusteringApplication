import unittest
import numpy as np
from main import parse_data  # Replace 'your_module' with the actual module name


class TestParseData(unittest.TestCase):

    def test_parse_data(self):
        # Case 1: Testing with text that produces 1D array
        text_content = "1.0\n2.0\n3.0\n"
        expected_output = np.array([1.0, 2.0, 3.0])
        self.assertTrue((parse_data(text_content) == expected_output).all())

        # Case 2: Testing with text that produces 2D array
        text_content = "1.0 2.0\n3.0 4.0\n"
        expected_output = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue((parse_data(text_content) == expected_output).all())

        # Case 3: Testing with text that has empty lines
        text_content = "\n1.0 2.0\n\n3.0 4.0\n"
        expected_output = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue((parse_data(text_content) == expected_output).all())

    def test_parse_data_with_invalid_data(self):
        # Case 4: Testing with text that contains non-numeric value
        text_content = "1.0 2.0\n3.0 abc\n"
        with self.assertRaises(ValueError):
            parse_data(text_content)

    def test_parse_data_with_no_values(self):
        # Case 5: Testing with text that contains empty string
        text_content = ""
        expected_output = np.array([])
        self.assertTrue((parse_data(text_content) == expected_output).all())


if __name__ == '__main__':
    unittest.main()