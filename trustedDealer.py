from multiplicationTriple import MultiplicationTriple

class TrustedDealer:
    def __init__(self, num_parties, bit_width=32):
        """
        Initialize the TrustedDealer class.

        :param num_parties: Number of parties involved in the secret sharing.
        :param bit_width: Bit width for the modulus.
        """
        self.num_parties = num_parties
        self.bit_width = bit_width
        self.modulus = 2 ** bit_width
        self.triples = []  # List of multiplication triples

    def generate_triple(self, num_triples):
        """
        Generate a specified number of multiplication triples.

        :param num_triples: Number of triples to generate.
        """
        for _ in range(num_triples):
            triple = MultiplicationTriple(self.num_parties, self.bit_width)
            self.triples.append(triple)

    def distribute_triples(self, triple_index):
        """
        Distribute the shares of a specified triple.

        :param triple_index: Index of the triple to distribute.
        :raises IndexError: If the triple index is out of range.
        """
        if triple_index >= len(self.triples):
            raise IndexError("Triple index out of range")
        triple = self.triples[triple_index]
        shares = triple.get_shares()
        return shares

    def get_number_of_triples(self):
        """
        Get the number of generated triples.

        :return: Number of triples.
        """
        return len(self.triples)