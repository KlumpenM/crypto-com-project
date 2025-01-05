import random
import numpy as np
from Crypto.Util import number
from phe import paillier
from sympy import mod_inverse
import secrets
import math
import gmpy2


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

def mult_triples(n, d, t, l, batch_size=None):
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
        Bit length of the elements in the matrices. Typically the key size

    Returns
    -------
    tuple
        the shared triplets [U], [V], [Z], [V'], [Z']
    """

    if batch_size == None:
        batch_size = int(np.floor(n / t))
    
    group = 2**l

    # Generate the random matrices U, V and V_prime
    random_state = gmpy2.random_state()
    U = np.array([[gmpy2.mpz_urandomb(random_state, l) for _ in range(d)] for _ in range(n)], dtype=object)
    V = np.array([[gmpy2.mpz_urandomb(random_state, l) for _ in range(t)] for _ in range(d)], dtype=object)
    V_prime = np.array([[gmpy2.mpz_urandomb(random_state, l) for _ in range(t)] for _ in range(batch_size)], dtype=object)
    #U = np.random.randint(low=2**l, high=None, size=(n, d))
    #V = np.random.randint(low=2**l, high=None, size=(d, t))
    #V_prime = np.random.randint(low=2**l, high=None, size=(batch_size, t))
    print("FUCK THE WORLD")

    # Directly computing the triplets without using offline phase.

    Z = U[0:batch_size,:] @ V[:,0:1]
    Z_prime = U[0:batch_size,:].transpose() @ V_prime[:,0:1]

    # Iterate over t mini-batches to compute Z and Z'
    for i in range(1, t):
        U_B_i = U[i*batch_size:i*batch_size+batch_size,]    # |B| x d
        V_i = V[:,i:i+1]                                    # d x t
        prod0 = U_B_i @ V_i
        Z = np.hstack((Z, prod0))                           # |B| x t

        U_B_i_T = U_B_i.transpose()                         # d x |B|
        V_prime_i = V_prime[:,i:i+1]                        # |B| x t
        prod1 = U_B_i_T @ V_prime_i
        Z_prime = np.hstack((Z_prime, prod1))               # d x t

    divisor = np.full(shape=Z.shape, fill_value=2**l)
    Z = np.mod(Z, divisor)

    divisor = np.full(shape=Z_prime.shape, fill_value=2**l)
    Z_prime = np.mod(Z_prime, divisor)

    """
    We now have the actual triplets used for multiplication, but we need to define the secret shares of U, V, Z, V' and Z' and distribute them
    among the parties.
    The Z and Z' we have just computed was not in a secure way. We will use them later to assert that the secure computation of Z and Z' is correct.
    """
    # Compute the shares of U, V and V'
    U0, U1 = share_matrix(U, l)
    V0, V1 = share_matrix(V, l)
    Vp0, Vp1 = share_matrix(V_prime, l)

    # The shares of Z and Z' are to be computed per column just as how MZ17 describes under section B. The Offline Phase in page 8.
    #   When computing the secret shares of the matrix product, it is supposedly here that truncation comes into play. The resulting product
    #   must be truncated down to some bit size.
    
    # LHE-based geneneration

    # Generate the keys of Paillier Cryptosystem
    pk, sk = paillier_keygen(2048)

    # Compute the shares of Z
    ## For the first column of Z
    A0 = []
    A0.append(U0[0:batch_size,:])                 # |B| x d
    B1 = []
    B1.append(V1[:,0:1])                          # d x 1
    A0B1 = LHE_MT(A0[0], B1[0], l, keys=(pk, sk)) # Is a tuple of shares of the product A0 x B1, each of shape |B| x 1

    A1 = []
    A1.append(U1[0:batch_size,:])                 # |B| x d
    B0 = []
    B0.append(V0[:,0:1])                          # d x 1
    A1B0 = LHE_MT(A1[0], B0[0], l, keys=(pk, sk)) # Is a tuple of shares of the product A1 x B0, each of shape |B| x 1

    ## For each of the remaining columns of Z
    for i in range(1,t):
        A0.append(U0[i*batch_size:i*batch_size+batch_size,:])
        B1.append(V1[:,i:i+1])
        C = LHE_MT(A0[i], B1[i], l, keys=(pk, sk))
        # We do the reshape because the vector is initially of shape (|B|,), but we want its shape to be (|B|,1)
        #   Otherwise, the hstack will not work.
        # But apparently, we do not need it after all????? Python just decided that np should stop doing it somehow?????
        new_var = np.hstack((A0B1[0], C[0])) # The first secret share of the product A0 x B1
        new_var1 = np.hstack((A0B1[1], C[1])) # The second secret share of the product A0 x B1
        A0B1 = (new_var, new_var1) # The secret shares of the product A0 x B1

        A1.append(U1[i*batch_size:i*batch_size+batch_size,:])
        B0.append(V0[:,i:i+1])
        C1 = LHE_MT(A1[i], B0[i], l, keys=(pk, sk)) # A secret share of a column
        new_var2 = np.hstack((A1B0[0], C1[0])) # The first secret share of the product A1 x B0
        new_var3 = np.hstack((A1B0[1], C1[1])) # The second secret share of the product A1 x B0
        A1B0 = (new_var2, new_var3) # The secret shares of the product A1 x B0
    
    # At this point, we have now computed the shares of Z

    # Next is to compute the shares of Z'
    A0_ = []
    A0_.append(U0[0:batch_size].transpose())
    B1_ = []
    B1_.append(Vp1[:,0:1])
    A0B1_ = LHE_MT(A0_[0], B1_[0], l, keys=(pk, sk)) # Is a tuple of shares of the product A0 x B1

    A1_ = []
    A1_.append(U1[0:batch_size,:].transpose())
    B0_ = []
    B0_.append(Vp0[:,0:1])
    A1B0_ = LHE_MT(A1_[0], B0_[0], l, keys=(pk, sk)) # Is a tuple of shares of the product A1 x B0

    for i in range(1,t):
        A0_.append(U0[i*batch_size:i*batch_size+batch_size,:].transpose())
        B1_.append(Vp1[:,i:i+1])
        C_ = LHE_MT(A0_[i], B1_[i], l, keys=(pk, sk))

        new_var = np.hstack((A0B1_[0], C_[0]))
        new_var1 = np.hstack((A0B1_[1], C_[1]))
        A0B1_ = (new_var, new_var1)

        A1_.append(U1[i*batch_size:i*batch_size+batch_size,:].transpose())
        B0_.append(Vp0[:,i:i+1])
        C1_ = LHE_MT(A1_[i], B0_[i], l, keys=(pk, sk))

        new_var2 = np.hstack((A1B0_[0], C1_[0]))
        new_var3 = np.hstack((A1B0_[1], C1_[1]))
        A1B0_ = (new_var2, new_var3)
    
    # By now, the shares of Z' has been computed

    # Testing if gathering the shares compute to Z
    assert U0.shape == U1.shape
    assert V0.shape == V0.shape

    # Remember that shape of Z is actually |B| x t
    for i in range(t):
        term1 = A0[i] @ B0[i]
        term2 = (A0B1[0][:,i:i+1] + A0B1[1][:,i:i+1])
        term3 = (A1B0[0][:,i:i+1] + A1B0[1][:,i:i+1])
        term4 = A1[i] @ B1[i]
        result = term1 + term2 + term3 + term4
        divisor = np.full(shape=result.shape, fill_value=2**l)
        result = np.mod(result, divisor)

        assert (result == Z[:,i:i+1]).all()
    
    # Time to define the secret shares of each column in Z
    term01, term11 = share_matrix(A0[0] @ B0[0], l) # Suppose that party 0 computes the secret shares of the matrix product and sends the share of party 1 to them.
    term02 = A0B1[0][:,0:1]
    term03 = A1B0[0][:,0:1]
    term04, term14 = share_matrix(A1[0] @ B1[0], l) # Suppose that party 1 computes the secret shares of the matrix product and sends the share of party 0 to them.

    # Share of party 0
    Z0 = term01 + term02 + term03 + term04 # With their share, party 0 computes the first column of Z

    term12 = A0B1[1][:,0:1]
    term13 = A1B0[1][:,0:1]

    # Share of party 1
    Z1 = term11 + term12 + term13 + term14

    # Ensure that the elements are in the group
    divisor = np.full(shape=Z0.shape, fill_value=2**l)
    Z0 = np.mod(Z0, divisor)
    Z1 = np.mod(Z1, divisor)
    
    assert (Z[:,0:1] == np.mod(Z0 + Z1, divisor)).all() # If this passes, then I may assume that it will pass as well for the remaining columns.

    for i in range(1,t):
        term01, term11 = share_matrix(A0[i] @ B0[i], l)
        term02 = A0B1[0][:,i:i+1]
        term03 = A1B0[0][:,i:i+1]
        term04, term14 = share_matrix(A1[i] @ B1[i], l)
        term12 = A0B1[1][:,i:i+1]
        term13 = A1B0[1][:,i:i+1]
        Z0_col = term01 + term02 + term03 + term04
        Z1_col = term11 + term12 + term13 + term14
        Z0_col = np.mod(Z0_col, divisor)
        Z1_col = np.mod(Z1_col, divisor)
        Z0 = np.hstack((Z0, Z0_col))
        Z1 = np.hstack((Z1, Z1_col))

    # Check if the whole matrix is correct
    divisor = np.full(shape=Z.shape, fill_value=2**l)
    assert (Z == np.mod(Z0 + Z1, divisor)).all()
    print('Shares of Z computed successfully')

    # TODO: Compute the shares of Z'
    term01_, term11_ = share_matrix(A0_[0] @ B0_[0], l)
    term02_ = A0B1_[0][:,0:1]
    term03_ = A1B0_[0][:,0:1]
    term04_, term14_ = share_matrix(A1_[0] @ B1_[0], l)

    # Share of party 0
    Z0_ = term01_ + term02_ + term03_ + term04_

    term12_ = A0B1_[1][:,0:1]
    term13_ = A1B0_[1][:,0:1]

    # Share of party 1
    Z1_ = term11_ + term12_ + term13_ + term14_

    # Ensure the elements are in the group
    divisor = np.full(shape=Z0_.shape, fill_value=2**l)
    Z0_ = np.mod(Z0_, divisor)
    Z1_ = np.mod(Z1_, divisor)

    for i in range(1,t):
        term01_, term11_ = share_matrix(A0_[i] @ B0_[i], l)
        term02_ = A0B1_[0][:,i:i+1]
        term03_ = A1B0_[0][:,i:i+1]
        term04_, term14_ = share_matrix(A1_[i] @ B1_[i], l)
        term12_ = A0B1_[1][:,i:i+1]
        term13_ = A1B0_[1][:,i:i+1]
        Z0_col_ = term01_ + term02_ + term03_ + term04_
        Z1_col_ = term11_ + term12_ + term13_ + term14_
        Z0_col_ = np.mod(Z0_col_, divisor)
        Z1_col_ = np.mod(Z1_col_, divisor)
        Z0_ = np.hstack((Z0_, Z0_col_))
        Z1_ = np.hstack((Z1_, Z1_col_))

    # Check if the whole matrix is correct
    divisor = np.full(shape=Z_prime.shape, fill_value=2**l)
    assert (Z_prime == np.mod(Z0_ + Z1_, divisor)).all()
    print('Shares of Z\' computed successfully')

    # Output the secret shares of the arithmetic multiplication triplets.
    return (U0, U1, V0, V1, Z0, Z1, Vp0, Vp1, Z0_, Z1_)

def share_matrix(M, l):
    """ Computes the secret shares of a matrix.
    
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
    r, c = M.shape
    
    random_state = gmpy2.random_state()
    M0 = np.array([[gmpy2.mpz_urandomb(random_state, l) for _ in range(c)] for _ in range(r)], dtype=object)
    #M0 = np.random.randint(low=2**l, high=None, size=(r, c))
    M1 = np.subtract(M, M0)
    divisor = np.full(shape=(r, c), fill_value=2**l)
    M1 = np.mod(M1, divisor)

    return M0, M1


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
        pk, sk = paillier_keygen(2048)
    else:
        pk, sk = keys

    # TODO: Implement Paillier encryption scheme... Or maybe even d-HE scheme, since we have done before in an assignment.
    #       But the paper refers to Pailler as an example.
    
    # Step 1
    #print('LHE_MT step 1')
    enc_B = []
    for i in range(B.shape[0]):
        #print(B[i,0])
        enc_Bi = paillier_enc(B[i,0], pk)    
        #print(f'enc_Bi: {enc_Bi}')
        enc_B.append(enc_Bi)
    
    assert len(enc_B) == B.shape[0]
    
    # Step 2
    #print('LHE_MT step 2')
    C = []
    r = np.empty(shape=(A.shape[0], 1))
    for i in range(A.shape[0]):
        r[i] = np.random.randint(0, (2**(l-1))/3)
    for i in range(A.shape[0]):
        big_prod = []
        # Big product sign
        for j in range(B.shape[0]):
            #print(f'enc_B[j]: {enc_B[j]}')
            #print(f'A[i,j]: {A[i,j].dtype}')
            print(f'j = {j}')
            big_prod.append(pow(enc_B[j], A[i,j]))
            #print(f'prod: {prod}')
        C.append(math.prod(big_prod) * paillier_enc(r[i], pk))
    
    # Step 3
    #print('LHE_MT step 3')
    r = r * -1
    divisor = np.full(shape=r.shape, fill_value=2**l)
    AB0 = np.mod(r, divisor)

    # Step 4
    #print('LHE_MT step 4')
    AB1 = np.empty(shape=(len(C),1))
    for i in range(len(C)):        
        AB1[i] = paillier_dec(C[i], pk, sk)

    return AB0, AB1


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

    # Use Chinese Remainder Theorem to find sk
    remainders = [0, 1]
    moduli = [(p-1)*(q-1), pk]
    sk, _ = solve_crt(remainders, moduli)

    return pk, sk


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
        while True:
            r = secrets.randbelow(pk - 1) + 1
            if math.gcd(r, pk) == 1:
                return r
    
    r = rand_r()
    #print('Found random coprime')

    m = gmpy2.mpz(m)
    pk = gmpy2.mpz(pk)
    alpha = (1 + m*pk) % pk**2
    beta = pow(r, pk, pk**2)
    ciphertext = (alpha * beta) % pk**2
    return ciphertext



def paillier_dec(c, pk, sk):
    """ The decryption scheme for Paillier.
    
    Parameters
    ----------
    c : int
        The ciphertext
    pk : int
        The public key
    sk : int
        The secret key
    
    Returns
    -------
    The plaintext
    """
    # Step 1: Compute c^sk mod pk^2
    c_sk = pow(c, sk, pk**2)

    # Step 2: Compute L function: L(x) = (x - 1) // pk
    L_c_sk = (c_sk - 1) // pk

    # Step 3: Compute the modular inverse of sk mod pk
    sk_inv = pow(sk, -1, pk)

    # Step 4: Recover plaintext
    m = (L_c_sk * sk_inv) % pk

    return m


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


def solve_crt(remainders, moduli):
    """
    Solve a system of congruences using the Chinese Remainder Theorem.
    
    x ≡ remainders[i] (mod moduli[i]) for all i
    
    :param remainders: List of remainders [r1, r2, ...]
    :param moduli: List of moduli [m1, m2, ...]
    :return: The solution x and the combined modulus
    """
    if len(remainders) != len(moduli):
        raise ValueError("Remainders and moduli must have the same length.")
    
    x = 0
    M = 1
    for m in moduli:
        M *= m  # Compute the product of all moduli
    
    for r, m in zip(remainders, moduli):
        Mi = M // m  # Partial modulus
        Mi_inverse = mod_inverse(Mi, m)  # Modular inverse of Mi mod m
        x += r * Mi * Mi_inverse  # Contribution to the solution
    
    return x % M, M