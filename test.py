from fixed_point_arithmetic import fixed_point_mult
import matplotlib.pyplot as plt
import multiplicationTriple
import numpy as np
from phe import paillier
import secrets

def test_mult_triple_gen():
    multiplicationTriple.mult_triples(8, 3, 4, 16)

def test_share_matrix():
    m = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    m0, m1 = multiplicationTriple.share_matrix(m, 3)
    print(f'm0: \n {m0}')
    print(f'm1: \n {m1}')
    m_prime = m0 + m1
    divisor = np.full(shape=m.shape, fill_value=2**3)
    print(f'divisor: \n {divisor}')
    m_prime = np.mod(m_prime, divisor)
    print(f'm_prime: \n {m_prime}')
    assert (m == m_prime).all()

def test_paillier_keygen():
    pk, sk = multiplicationTriple.paillier_keygen(2048)
    print(f'pk: {pk}')
    print(f'sk: {sk}')

def test_paillier_cryptosystem():
    key_len = 2048
    print('Generating keys')
    pk, sk = multiplicationTriple.paillier_keygen(key_len)
    print('Keys are generated')
    message = secrets.randbits(key_len)
    print('Random message chosen')
    ciphertext = multiplicationTriple.paillier_enc(message, pk)
    print('Ciphertext computed')
    message1 = multiplicationTriple.paillier_dec(ciphertext, pk, sk)
    print('Plaintext computed')
    assert message == message1, f'message: {message} \n message1: {message1}'

def numpy_sucks():
    b = np.ones((10,))
    print(f'b shape: {b.shape}')
    new_b = b.reshape(-1,1)
    print(f'b new shape: {new_b.shape}')

def matrix_sanity_check():
    A = np.array([[1, 2],
                  [3, 4],
                  [5,6]])
    B = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    
    assert (A @ B == np.dot(A, B)).all()

def test_simple_fixed_point_mult():
    x = 1.25
    y = 2.5
    l = 2
    z = fixed_point_mult(x, y, l)
    print(z)
    print(x * y)

def test_fixed_point_mult_plot_over_bitlen():
    l = np.arange(1, 21)
    x = 1.25
    y = 2.5
    z = np.zeros(shape=l.shape)
    for i in range(len(l)):
        z[i] = fixed_point_mult(x, y, l[i])
    
    # See how fixed point multiplication errs from the normal multiplication.
    error = z - np.full(shape=z.shape, fill_value=x * y)
    plt.plot(l, error)
    plt.show()

if __name__ == "__main__":
    #test_mult_triple_gen()
    #test_share_matrix()
    #test_paillier_keygen()
    #test_paillier_cryptosystem()
    #matrix_sanity_check()
    #test_simple_fixed_point_mult()
    test_fixed_point_mult_plot_over_bitlen()