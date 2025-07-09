import os
import openai
import json
import evaluate
from datasets import load_dataset

# --- CAMEL의 RoleType, BaseMessage 컨셉만 가져옵니다. ---
from camel.types import RoleType
from camel.messages import BaseMessage

# --- vLLM 서버 연동을 위한 환경 변수 설정 (필수!) ---
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000"
os.environ["OPENAI_API_KEY"] = "sk-dummy"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# --- openai 클라이언트 초기화 ---
client = openai.OpenAI()

# --- Custom Agent 클래스 정의 (CAMEL의 ChatAgent 역할) ---
class CustomAgent:
    def __init__(self, role_name: str, role_type: RoleType, model_name: str = "Qwen/Qwen3-1.7B"):
        self.role_name = role_name
        self.role_type = role_type
        self.model_name = model_name
        self.history = []

        system_content = f"You are a helpful {self.role_name}. Your goal is to assist in solving programming problems."
        if self.role_type == RoleType.USER:
            system_content += " You will act as a programmer, writing code based on requirements."
        elif self.role_type == RoleType.ASSISTANT:
            system_content += " You will act as a code reviewer, providing constructive feedback on code."
            
        self.history.append({"role": "system", "content": system_content})

    def step(self, input_message: BaseMessage) -> BaseMessage:
        messages_for_llm = []
        for msg in self.history:
            messages_for_llm.append({"role": msg["role"], "content": msg["content"]})
        messages_for_llm.append({"role": "user", "content": input_message.content})

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_llm,
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )

            llm_response_content = response.choices[0].message.content
            
            self.history.append({"role": "user", "content": input_message.content})
            self.history.append({"role": "assistant", "content": llm_response_content})

            return BaseMessage(
                role_name=self.role_name,
                role_type=self.role_type,
                content=llm_response_content,
                meta_dict={}
            )

        except openai.APIConnectionError as e:
            print(f"API Connection Error: Could not connect to vLLM server. {e}")
            raise
        except openai.APIStatusError as e:
            print(f"API Status Error: Status {e.status_code}, Response {e.response}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            raise

# --- Custom Agent 인스턴스 생성 ---
programmer_agent = CustomAgent("Python Programmer", RoleType.USER, model_name="Qwen/Qwen3-1.7B")
tester_agent = CustomAgent("Python Tester", RoleType.ASSISTANT, model_name="Qwen/Qwen3-1.7B")

# --- 코드 추출 함수 (이전과 동일) ---
def extract_code_from_response(response_content: str) -> str:
    start_marker = "```python"
    end_marker = "```"
    start_index = response_content.find(start_marker)
    if start_index == -1: return ""
    code_start = start_index + len(start_marker)
    code_end = response_content.find(end_marker, code_start)
    if code_end == -1: return response_content[code_start:].strip()
    else: return response_content[code_start:code_end].strip()

# --- MAS 대화 실행 함수 ---
def run_mas_for_humaneval_custom(
    problem_prompt: str,
    max_turns: int = 5
) -> str:
    print(f"\n--- Starting Custom MAS conversation for problem ---\n")
    
    initial_prompt_to_programmer = BaseMessage(
        role_name="user",
        role_type=RoleType.USER,
        content=f"You are a Python programmer. Your task is to write a Python function that solves the following problem. "
                 f"Provide only the function code within a ```python``` block. Do not include explanations, examples, or extra text.\n\n"
                 f"Problem:\n{problem_prompt}",
        meta_dict={}
    )
    
    current_message = initial_prompt_to_programmer
    generated_code = ""

    for i in range(max_turns):
        print(f"Turn {i+1}:")
        
        try:
            if current_message.role_type == RoleType.USER:
                print(f"  Programmer receives: {current_message.content[:80]}...")
                response_from_programmer = programmer_agent.step(current_message)
                current_message = response_from_programmer
                
                extracted_code = extract_code_from_response(current_message.content)
                if extracted_code:
                    generated_code = extracted_code
                    print(f"  Programmer generated code (first 80 chars):\n  {generated_code[:80]}...")
                    
                    feedback_request_to_tester = BaseMessage(
                        role_name="user",
                        role_type=RoleType.USER,
                        content=f"You are a Python tester. Review the following code for correctness, efficiency, and adherence to the problem. "
                                 f"Provide constructive feedback or state 'CODE_FINALIZED' if it looks good.\n\n"
                                 f"```python\n{generated_code}\n```",
                        meta_dict={}
                    )
                    current_message = feedback_request_to_tester
                else:
                    print(f"  Programmer did not provide valid code. Requesting again.")
                    re_request_to_programmer = BaseMessage(
                        role_name="user",
                        role_type=RoleType.USER,
                        content="Please provide the complete Python function code in a ```python``` block, and nothing else.",
                        meta_dict={}
                    )
                    current_message = re_request_to_programmer

            elif current_message.role_type == RoleType.ASSISTANT:
                print(f"  Tester receives: {current_message.content[:80]}...")
                response_from_tester = tester_agent.step(current_message)
                current_message = response_from_tester

                if "CODE_FINALIZED" in current_message.content.upper():
                    print("  Tester finalized the code. Ending conversation.")
                    break
                else:
                    print(f"  Tester provided feedback: {current_message.content[:80]}...")
                    feedback_to_programmer = BaseMessage(
                        role_name="user",
                        role_type=RoleType.USER,
                        content=f"The tester provided the following feedback. Please revise your code based on this feedback and provide the complete, updated function within a ```python``` block, and nothing else.\n\n"
                                 f"Feedback:\n{current_message.content}",
                        meta_dict={}
                    )
                    current_message = feedback_to_programmer
            
            else:
                print(f"Unknown role type: {current_message.role_type}. Ending conversation.")
                break

        except openai.APIConnectionError as e:
            print(f"API Connection Error: Could not connect to the vLLM server. Is it running? {e}")
            break
        except openai.APIStatusError as e:
            print(f"API Status Error: Status {e.status_code}, Response {e.response}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during agent interaction: {e}")
            break

    print(f"\n--- Custom MAS Conversation Ended. Final Code: ---\n{generated_code[:200]}...")
    return generated_code

# --- HumanEval 평가 파이프라인 ---
def run_humaneval_pipeline_custom(num_problems: int = 5):
    dataset = load_dataset("bigcode/humanevalpack", "python")
    problems = dataset["test"] 

    predictions = []
    references_for_eval = []
    # entry_points_for_eval 와 problem_descriptions_for_eval은 이제 compute()에 직접 전달되지 않으므로 필요 없습니다.
    # problem_descriptions_for_eval = [] 

    print(f"\n===== Starting HumanEval Custom MAS Evaluation for {num_problems} problems =====")

    for i, problem in enumerate(problems):
        if i >= num_problems:
            break

        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test_code = problem["test"] # references_for_eval 에 사용
        # entry_point = problem["entry_point"] # 더 이상 직접 사용되지 않습니다.

        print(f"\nProcessing {task_id} (Prompt: {prompt[:50]}...)...") # 엔트리포인트 대신 프롬프트 일부 출력
        
        generated_code = run_mas_for_humaneval_custom(prompt, max_turns=5)

        if not generated_code:
            print(f"WARNING: No code generated for {task_id}. Using placeholder.")
            generated_code = "def placeholder_func(): pass # No code generated by MAS"
            
        predictions.append([generated_code])
        references_for_eval.append(test_code)
        # entry_points_for_eval.append(entry_point) # 더 이상 사용되지 않습니다.
        # problem_descriptions_for_eval.append(prompt) # 더 이상 사용되지 않습니다.

    print("\n===== Evaluation Complete. Calculating Metrics =====")

    humaneval_metric = evaluate.load("code_eval")
    
    results = humaneval_metric.compute(
        predictions=predictions,
        references=references_for_eval
        # entry_points와 problem_description은 이 버전의 code_eval에서 직접적인 키워드 인자가 아닙니다.
        # code_eval 메트릭은 references 문자열 내에서 테스트 케이스 및 엔트리 포인트를 자체적으로 처리할 것으로 예상됩니다.
    )

    print("\n--- HumanEval Evaluation Results ---")
    print(f"Pass@1: {results[0]['pass@1']:.4f}")

    return results

# --- 파이프라인 실행 ---
if __name__ == "__main__":
    print("Starting HumanEval Custom MAS Testing Pipeline...")
    # vLLM 서버가 실행 중인지 확인하세요! (예: python -m vllm.entrypoints.openai.api_server --model "Qwen/Qwen3-1.7B" --port 8000 --host 127.0.0.1)

    final_results = run_humaneval_pipeline_custom(num_problems=50)
    print("\nFinal Evaluation Results Summary:")
    print(final_results)