from human_eval.data import read_problems

def load_problems(n=1):
    problems = read_problems()
    problem_keys = list(problems)[:n]
    return problem_keys, problems  # 원하는 개수만
