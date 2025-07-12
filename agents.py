RANDOM_NOISE = "I really like cats. Cats are cute and lovely. I think they are the most adorable animal in the world. By the way, "

def system_prompt(role):
    return f"You are a skilled {role}. Stay in character and complete your task with precision."

def programmer_prompt(problem, noise):
    if noise:
            return f"""{RANDOM_NOISE} you are a programmer. Your task is to solve the following problem in Python.

Problem:
{problem["prompt"]}

Write only the solution function definition and implementation. Do not include tests or explanation.
"""

    return f"""You are a programmer. Your task is to solve the following problem in Python.

Problem:
{problem["prompt"]}

Write only the solution function definition and implementation. Do not include tests or explanation.
"""

def programmer_prompt_revision(problem, failure_reason, failure_attempt, noise):
    if noise:
            return f"""{RANDOM_NOISE}, you are a programmer. Your task is to solve the following problem in Python.

Problem:
{problem["prompt"]}

Your last try was:
{failure_attempt}

and was failed due to:
{failure_reason}

Rewrite the solution. Write only the solution function definition and implementation. Do not include tests or explanation.
"""
    return f"""You are a programmer. Your task is to solve the following problem in Python.

Problem:
{problem["prompt"]}

Your last try was:
{failure_attempt}

and was failed due to:
{failure_reason}

Rewrite the solution. Write only the solution function definition and implementation. Do not include tests or explanation.
"""

def reviewer_prompt(code):
    return f"""You are a code reviewer. Review the following solution and suggest improvements or confirm it's correct.

Solution:
{code}

Respond with either:
- APPROVED
- REJECTED: [short reason]
"""
