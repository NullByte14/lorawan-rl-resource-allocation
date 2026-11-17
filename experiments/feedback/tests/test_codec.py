"""Round-trip and corruption checks for the actual compact on-air payload."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback_codec import decode_feedback, encode_feedback, self_check


class CodecTests(unittest.TestCase):
    def test_all_receipt_patterns_and_reserved_fields(self):
        result = self_check()
        self.assertEqual(result["bitmap_patterns"], 65536)
        self.assertEqual(result["snr_codes"], 56)
        self.assertEqual(result["one_byte_patterns"], 1)
        self.assertEqual(result["two_byte_patterns"], 16)
        self.assertEqual(result["three_byte_patterns"], 65519)
        self.assertEqual(result["malformed_payloads_rejected"], 36)

    def test_payload_contains_individual_outcomes_not_just_a_count(self):
        first = (False, True) + (True,) * 14
        second = (True, False) + (True,) * 14
        self.assertNotEqual(encode_feedback(first, 12), encode_feedback(second, 12))
        self.assertEqual(decode_feedback(encode_feedback(first, 12)).receipts, first)
        self.assertEqual(decode_feedback(encode_feedback(second, 12)).receipts, second)

    def test_unknown_snr_bin_is_a_distinct_valid_state(self):
        self.assertEqual(decode_feedback(encode_feedback((True,)*16, 55)).snr_bin, 55)
        with self.assertRaises(ValueError):
            decode_feedback(bytes([56]))


if __name__ == "__main__":
    unittest.main()
