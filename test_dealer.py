import unittest
from dealer import generate_mac_key, generate_homomorphic_mac_key, generate_mac_tag, verify_mac_tag, verify_homomorphic_mac_tag

# test_dealer.py


class TestDealer(unittest.TestCase):

    def test_generate_mac_key(self):
        prime = 101
        key = generate_mac_key(prime)
        self.assertEqual(len(key), 2)
        self.assertTrue(0 <= key[0] < prime)
        self.assertTrue(0 <= key[1] < prime)

    def test_generate_homomorphic_mac_key(self):
        prime = 101
        key_suffix = 50
        key = generate_homomorphic_mac_key(prime, key_suffix)
        self.assertEqual(len(key), 2)
        self.assertTrue(0 <= key[0] < prime)
        self.assertTrue(0 <= key[1] < key_suffix)

    def test_generate_mac_tag(self):
        key = (5, 3)
        message = 10
        tag = generate_mac_tag(key, message)
        self.assertEqual(tag, 5 * 10 + 3)

    def test_verify_mac_tag_success(self):
        key = (5, 3)
        message = 10
        tag = generate_mac_tag(key, message)
        self.assertTrue(verify_mac_tag(key, tag, message))

    def test_verify_mac_tag_failure(self):
        key = (5, 3)
        message = 10
        tag = generate_mac_tag(key, message) + 1
        self.assertFalse(verify_mac_tag(key, tag, message))

    def test_verify_homomorphic_mac_tag_success(self):
        keys = [(5, 3), (7, 2)]
        messages = [10, 20]
        tags = [generate_mac_tag(keys[0], messages[0]), generate_mac_tag(keys[1], messages[1])]
        self.assertTrue(verify_homomorphic_mac_tag(tags, keys, messages))

    def test_verify_homomorphic_mac_tag_failure(self):
        keys = [(5, 3), (7, 2)]
        messages = [10, 20]
        tags = [generate_mac_tag(keys[0], messages[0]), generate_mac_tag(keys[1], messages[1]) + 1]
        self.assertFalse(verify_homomorphic_mac_tag(tags, keys, messages))

if __name__ == '__main__':
    unittest.main()