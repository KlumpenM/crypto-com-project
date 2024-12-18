import random
import numpy as np
from Crypto.Util import number
from phe import paillier


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
        Number of columns/features
    t : int
        Number of mini-batches
    l : int
        Bit length

    Returns
    -------
    tuple
        the shared triplets [U], [V], [Z], [V'], [Z']
    """

    batch_size = int(np.floor(n / t))

    # Generate the random matrices U, V and V_prime
    # The upperbound of each element is (2**(l-1))/3 because that is the limit of what Paillier can encrypt.
    U = np.random.randint((2**(l-1))/3, size=(n, d))
    #print(f'U: \n {U}')
    V = np.random.randint((2**(l-1))/3, size=(d, t))
    #print(f'V: \n {V}')
    V_prime = np.random.randint((2**(l-1))/3, size=(batch_size, t))

    #print(f'Mini-batch size: {batch_size}')

    #print(U[0:batch_size,:].shape)
    #print(V[:,0].shape)

    Z = U[0:batch_size,:] @ V[:,0:1]
    #print(Z.shape)
    #print(Z)

    Z_prime = U[0:batch_size,:].transpose() @ V_prime[:,0:1]

    # Iterate over t mini-batches to compute Z and Z'
    for i in range(1, t):
        U_B_i = U[i*batch_size:i*batch_size+batch_size,]    # |B| x d
        #print(f'Submatrix of U: \n {U_B_i}')
        V_i = V[:,i:i+1]                                    # d x t
        Z = np.hstack((Z, U_B_i @ V_i))                     # |B| x t

        U_B_i_T = U_B_i.transpose()                         # d x |B|
        V_prime_i = V_prime[:,0:1]                          # |B| x t
        Z_prime = np.hstack((Z_prime, U_B_i_T @ V_prime_i)) # d x t

    #print(f'Shape of Z: {Z.shape}')
    #print(f'Z: \n {Z}')
    #print(f'Shape of Z\': {Z_prime.shape}')
    #print(f'Z\': \n {Z_prime}')

    """
    We now have the actual triplets used for multiplication, but we need to define the secret shares of U, V, Z, V' and Z' and distribute them
    among the parties.
    """
    # TODO: Compute the shares of the triplets
    U0, U1 = share_matrix(U, l)
    V0, V1 = share_matrix(V, l)
    Vp0, Vp1 = share_matrix(V_prime, l)

    # The shares of Z and Z' are to be computed per column just as how MZ17 describes under section B. The Offline Phase in page 8
    # TODO: insert either algorithm for LHE-based gen or OT-based gen

    # Generate the keys of Paillier Cryptosystem
    pk, sk = paillier.generate_paillier_keypair(n_length=l)

    A0 = U0[0:batch_size,:]
    B1 = V1[:,0:1]
    AB0 = LHE_MT(A0, B1, l, keys=(pk, sk))

    A1 = U1[0:batch_size,:]
    B0 = V0[:,0:1]
    AB1 = LHE_MT(A1, B0, l, key=(pk, sk))

    # TODO: Now repeat the above for the remaining columns




def share_matrix(M, l):
    """ Computes the secret shares of a matrix. Not sure if it should be defined here or somewhere else like the dealer for instance.
    
    Parameters
    ----------
    M : 2darray
        The matrix to secret share
    l : int
        Bit length to define the group Z_(2^l) which we sample from when computing the secret share
        
    Returns
    -------
    Tuple of matrices of same shape as M, also being the secret shares of M
    """

    #print(f'l: {l}')

    r, c = M.shape
    #print(2**l - 1)
    M0 = np.random.randint((2**(l-1))/3, size=(r, c))
    M1 = np.subtract(M, M0)
    divisor = np.full(shape=(r, c), fill_value=(2**(l-1))/3)
    M1 = np.mod(M1, divisor)

    return M0, M1

# TODO
def LHE_MT(A, B, l, keys=None):
    """ Based on MZ17 figure 12, the Offline Protocol based on Lineary Homomorphic Encryption

    Parameters
    ----------
    A : 2darray
        A |B| x d matrix share from one party
    B : 2darray
        A d x 1 matrix share from the other party
    l : int
        Bit length
    
    Returns
    -------
    A tuple of the shares of the product A x B
    """

    if keys == None:
        pk, sk = paillier.generate_paillier_keypair(n_length=l)
    else:
        pk, sk = keys

    # TODO: Implement Paillier encryption scheme... Or maybe even d-HE scheme, since we have done before in an assignment.
    #       But the paper refers to Pailler as an example.
    
    # Step 1
    enc_B = np.empty(shape=B.shape)
    for i in range(B.shape[0]):
        #print(B[i,0])
        enc_Bi = pk.encrypt(int(B[i,0]))    
        # Encryption output is an EncryptedNumber object. We would rather work with the actual value it represents
        #   which is accessed via EncryptedNumber.ciphertext(). Thus we may perform operations on the ciphertext.
        print(f'enc_Bi: {enc_Bi}')
        enc_B[i,0] = enc_Bi.ciphertext()
    
    # Step 2
    C = np.empty(shape=(A.shape[0], 1))
    r = np.empty(shape=(A.shape[0], 1))
    for i in range(A.shape[0]):
        r[i] = np.random.randint(0, (2**(l-1))/3)
    for i in range(A.shape[0]):
        prod = 1
        for j in range(B.shape[0]):
            prod *= enc_B[j]**A[i,j]
            print(f'prod: {prod}')
        C[i] = prod * pk.encrypt(int(r[i])).ciphertext()
    
    # Step 3
    r = r * -1
    divisor = np.full(shape=r.shape, fill_value=2**l)
    AB0 = np.mod(r, divisor)

    # Step 4
    AB1 = np.empty(shape=C.shape)
    for i in range(C.shape[0]):
        new_var = C[i,0]
        print(new_var)
        encrypted_number = paillier.EncryptedNumber(pk, new_var)
        AB1[i] = sk.decrypt(encrypted_number)

    return AB0, AB1

# Deprecated: use the phe Python package instead
def paillier_keygen(l):
    """ Generates the public and secret keys for the Paillier Cryptoscheme
    
    Parameters
    ----------
    l : int
        Bit length of keys

    Returns
    -------
    Public and secret key pair as tuple
    """
    
    p = number.getPrime(l)
    q = number.getPrime(l)

    pk = p * q

    # TODO: Use Chinese Remainde Theorem to find sk
    pass

# Deprecated: use the phe Python package instead
def paillier_enc(m, pk):
    """ The encryption scheme of Paillier. In context of MZ17, used to encrypt each element of a matrix.

    Parameters
    ----------
    m : int
        The input message/element
    pk : int
        The public key for encryption

    Returns
    -------
    The encryption of m
    """

    def rand_r():
        """ Helper function to sample a random r in Z*_N

        Returns
        -------
        Random r in Z*_N
        """
        r = np.random.randint(1, pk)
        if coprime(r, pk):
            return r
        else:
            return rand_r()
    
    r = rand_r()

    return np.mod((1 + m*pk) * r**pk, pk**2)



# Python3 program to check if two 
# numbers are co-prime or not

# Recursive function to
# return gcd of a and b
def __gcd(a, b):

    # Everything divides 0 
    if (a == 0 or b == 0): return 0
    
    # base case
    if (a == b): return a
    
    # a is greater
    if (a > b): 
        return __gcd(a - b, b)
            
    return __gcd(a, b - a)

# Function to check and print if 
# two numbers are co-prime or not 
def coprime(a, b):
    return __gcd(a, b) == 1