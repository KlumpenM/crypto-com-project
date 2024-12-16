import random
import numpy as np

class MultiplicationTriple:
    def __init__(self, num_parties, bit_width) -> None:
        """
        Initialize the MultiplicationTriple class.

        :param num_parties: Number of parties involved in the secret sharing.
        :param bit_width: Bit width for the modulus.
        """
        self.num_parties = num_parties
        self.bit_width = bit_width
        self.modulus = 2 ** bit_width

        # Initialize the shares
        self.a_shares = [random.randint(0, self.modulus - 1) for _ in range(num_parties - 1)]
        self.b_shares = [random.randint(0, self.modulus - 1) for _ in range(num_parties - 1)]

        # Compute the product of the shares
        a = sum(self.a_shares) % self.modulus
        b = sum(self.b_shares) % self.modulus

        # Compute c = a * b mod 2^k
        c = (a * b) % self.modulus
        
        # Secret share c
        self.c_shares = [random.randint(0, self.modulus - 1) for _ in range(num_parties - 1)]
        last_c_share = (c - sum(self.c_shares)) % self.modulus
        self.c_shares.append(last_c_share)

    def get_shares(self):
        """
        Get the shares of a, b, and c.

        :return: A list of tuples containing the shares of a, b, and c.
        """
        return list(zip(self.a_shares, self.b_shares, self.c_shares))
    

"""
Relevant resources:
- MZ17 page 6 under 'Vectorization in the Shared Setting' written in bold and down to page 7 until 'Arithmetic Operations on Shared Decimal Numbers' written in bold
- MZ17 page 8

Multiplication triples serves the purpose of computing the multiplication of two matrices A and B in a secret sharing. The triples are U, V and Z.
In other words, two parties wants to compute [A] x [B] that are secret shared. Thus a dealer must compute [U], [V] and [Z], where U has same dim as A and
V has same dim as B.

According to page 8 of MZ17, I deduce that matrix U has dimensions n x d, where n must be the number of samples and d must be the number of features. 
I can also deduce that the matrix V has d rows, but number of columns is unknown.
Elements of U and V are uniformly random in the group Z_{2^l}
A |B| x d submatrix of U is defined and called A, where |B| is the number of batches for stochastic gradient descent.
Finally, the paper aims to find the shares of C = A x B and repeat it such that they compute Z. A, B and Z are equivalent to the triples u,v and w from BeDOZa respectively.

There are two ways to compute the multiplication triples which MZ17 calls the offline phase: linear homomorphic encryption or oblivious transfer.
"""

"""
Linear homomorphic encryption
1. Generate random U and random V
2. Take a submatrix A out of U (random???)
3. Take a column B (Bad naming, but this is how MZ17 names them) out of V (random???)
4. Compute C such that C = [A]_0 x [B]_0 + [A]_0 x [B]_1 + [A]_1 x [B]_0 + [A]_1 x [B]_1
5. Compute [A]_0 x [B]_1
6. Compute [A]_1 x [B]_0
7. Remaining terms are computed locally without communication between parties.
"""

def mult_triples(n, d, t, l):
    """ Computes the multiplication triplets
    
    Parameters
    ----------
    n : int
        Number of rows/data samples
    d : int
        Number of columns/
    t : int
        Number of mini-batches
    l : int
        Bit length

    Returns
    -------
    tuple
        the shared triplets [U], [V], [Z], [V'], [Z']
    """

    U = np.random.randint(2^l - 1, size=(n, d))
    print(f'U: \n {U}')
    V = np.random.randint(2^l - 1, size=(d, t))
    print(f'V: \n {V}')
    
