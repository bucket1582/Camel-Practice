import openai
from humaneval_loader import load_problems
from evaluator import run_test_with_code
from agents import programmer_prompt, reviewer_prompt, system_prompt, programmer_prompt_revision
import time

openai.api_key = "EMPTY"  # 필요 없음
openai.api_base = "http://localhost:8000/v1"

# === vLLM Chat 요청 ===
def chat(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        messages=messages,
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message["content"].strip()

# === CAMEL + HumanEval 평가 ===
def run_camel_vllm(n=5, noise=False):
    problem_keys, problems = load_problems(n)
    results = []

    for idx, problem_key in enumerate(problem_keys):
        problem = problems[problem_key]
        print(f"\n=== Problem {idx+1}/{n} | ID: {problem['task_id']} ===")

        # Programmer 역할
        sys_p = system_prompt("programmer")
        prompt_p = programmer_prompt(problem, noise)
        code = chat(sys_p, prompt_p)
        # print(f"\n[PROGRAMMER CODE]\n{code}")

        MAX_TRIES = 3
        for _ in range(MAX_TRIES):
            # Reviewer 역할
            sys_r = system_prompt("reviewer")
            prompt_r = reviewer_prompt(code)
            review = chat(sys_r, prompt_r)
            # print(f"\n[REVIEWER RESPONSE]\n{review}")

            if "APPROVED" in review.upper():
                break
            else:
                new_prompt_p = programmer_prompt_revision(problem, review, code, noise)
                code = chat(sys_p, new_prompt_p)

        # 코드 실행
        passed, output = run_test_with_code(problem, code)
        print(f"[EXECUTION RESULT] {'✅ PASS' if passed else '❌ FAIL'}")
        results.append(passed)

    # 통계 출력
    total = len(results)
    passed = sum(results)
    print("\n=== 📊 Pass@1 Summary ===")
    print(f"Total: {total}, Passed: {passed}, Pass@1: {passed/total:.2%}")

if __name__ == "__main__":
    run_camel_vllm(n=100)
    run_camel_vllm(n=100, noise=True)
