class Party:
    def __init__(self, party_id, num_parties, bit_width=32):
        self.party_id = party_id # Index starting from 0
        self.num_parties = num_parties
        self.bit_width = bit_width
        self.modulus = 2 ** bit_width
        
        # Initialize the shares
        self.x_share = 0
        self.weight_share = 0
        self.triples = [] # List of (a_i, b_i, c_i) for each triple
        # Initialize a list to store partial multiplication results
        self.multiplications = []

    def set_input_share(self, x_share):
        self.x_share = x_share % self.modulus

    def set_weight_share(self, weight_share):
        self.weight_share = weight_share % self.modulus

    def receive_triple_shares(self, a_i, b_i, c_i):
        self.triples.append((a_i % self.modulus, b_i % self.modulus, c_i % self.modulus))

    def add(self, other_share):
        return (self.x_share + other_share) % self.modulus
    
    def secure_multiply(self, other_share, triple_index):
        if triple_index >= len(self.triples):
            raise IndexError("Triple index out of range")
        
        a_i, b_i, c_i = self.triples[triple_index]
        # Compute d_i and e_i
        d_i = (self.x_share - a_i) % self.modulus
        e_i = (other_share - b_i) % self.modulus

        # Each party broadcasts d_i and e_i to all other parties
        # In this class-based implementation, we'll assume that all parties have access to d and e
        # For simulation purposes, we'll gather d and e from all parties externally
        return (c_i + d_i * b_i + e_i * a_i) % self.modulus # Simplified without d * e term

