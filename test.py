import multiplicationTriple
import numpy as np

def test_mult_triple_gen():
    multiplicationTriple.mult_triples(8, 3, 4, 32)

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

if __name__ == "__main__":
    test_mult_triple_gen()
    test_share_matrix()