from QUBOConvert import SetCover
from QUBOConvert.utils import solve_qubo_bruteforce
    

U = {"a", "b", "c", "d"}
V = [{"a", "b"}, {"a", "c"}, {"c", "d"}]
problem = SetCover(U, V)


def test_setcover_logtrick_solve():
    
    Q, offset = problem.to_qubo()
    e, sol = solve_qubo_bruteforce(Q)
    solution = problem.convert_solution(sol)
    assert (
        problem.is_solution_valid(solution) and 
        solution == {0, 2} and
        e+offset == len(solution)
    )
    
    
def test_setcover_solve():
    
    Q_notlog, offset = problem.to_qubo(log_trick=False)
    e, sol = solve_qubo_bruteforce(Q_notlog)
    solution = problem.convert_solution(sol)
    assert (
        problem.is_solution_valid(solution) and 
        solution == {0, 2} and
        e+offset == len(solution)
    )
    

def test_setcover_logtrick_numvars():
    
    Q, offset = problem.to_qubo()
    assert (
        len(set(y for x in Q for y in x)) == problem.num_binary_variables()
    )
    
    
def test_setcover_numvars():
    
    Q_notlog, offset = problem.to_qubo(log_trick=False)
    assert (
        len(set(y for x in Q_notlog for y in x)) == 
        problem.num_binary_variables(False)
    )
    
    
def test_setcover_str():
    
    assert eval(str(problem)) == problem
    