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
    divisor = np.full(shape=m.shape, fill_value=2**3 - 1)
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

if __name__ == "__main__":
    test_mult_triple_gen()
    #test_share_matrix()
    #test_paillier_keygen()
    #test_paillier_cryptosystem()