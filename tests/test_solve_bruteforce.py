from qubovert.utils import solve_qubo_bruteforce, solve_ising_bruteforce


def test_solve_qubo_bruteforce():

    Q = {(0, 1): 1, (1, 2): 1, (1, 1): -1, (2, 2): -2}
    assert solve_qubo_bruteforce(Q) == (-2, [0, 0, 1])

    Q = {(0, 0): 1, (0, 1): -1}
    assert solve_qubo_bruteforce(Q, 1, True) == (1, [[0, 0], [0, 1], [1, 1]])


def test_solve_ising_bruteforce():

    h = {1: -1, 2: -2}
    J = {(0, 1): 1, (1, 2): 1}
    assert solve_ising_bruteforce(h, J) == (-3, [-1, 1, 1])

    h, J, offset = {0: 0.25, 1: -0.25}, {(0, 1): -0.25}, 1.25
    assert (
        solve_ising_bruteforce(h, J, offset, True)
        ==
        (1, [[-1, -1], [-1, 1], [1, 1]])
    )
