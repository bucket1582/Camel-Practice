import subprocess
import tempfile
import os

def run_test_with_code(problem, code: str, timeout=3):
    # Wrap function + test into a temporary script
    code = code[9:-3]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + problem["test"])
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        passed = result.returncode == 0
        output = result.stdout.decode() + result.stderr.decode()
    except subprocess.TimeoutExpired:
        passed = False
        output = "TIMEOUT"

    os.remove(temp_path)
    return passed, output
