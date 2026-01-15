import re
import requests
from transformers import AutoTokenizer
from typing import List, Dict, Optional, final, Union
from vllm import LLM

def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'http://localhost:1243/search'
    data = {'query': query, 'top_n': 5}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\n\n"
    retrieval_text = retrieval_text.strip()
    return retrieval_text

def batch_search(query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'http://localhost:1243/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)

    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            curr_result += f"{line['contents']}\n\n"
        result_list.append(curr_result.strip())

    return result_list


def extract_search_content(text: str) -> str:
    try:
        start_tag = '<search>'
        end_tag = '</search>'
        # assert text.strip().endswith(end_tag)
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""

def validate_template_format(text: str) -> tuple[bool, str]:
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> unpair"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "less <think> or </think> label"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "the appearance time for <answer> or </answer> is not 1"

    # 检查 search/result 标签的顺序
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result is uncomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result order error"

        current_pos = result_end_pos

    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must exist before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer needs \\boxed{}"

    return True, "correct format"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def extract_python_content(text: str) -> str:
    try:
        start_tag = '<python>'
        end_tag = '</python>'
        assert text.strip().endswith(end_tag)
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""


def last_boxed_only_string(string):
    try:
        idx = string.rfind("\\boxed")
    except:
        import pdb; pdb.set_trace()
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string[idx:]
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_answer(solution_str) -> float:
    answer_part = solution_str
    if "<answer>" in answer_part:
        answer_part = extract_answer(answer_part)
        if answer_part is not None:
            if "\\boxed{" in answer_part:
                answer = remove_boxed(last_boxed_only_string(answer_part))
            else:
                answer = answer_part
        else:
            answer = solution_str
            print("can not extract answer")

    elif "<answer>" not in answer_part:
        if "\\boxed{" in answer_part:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        else:
            answer = answer_part

    else:
        answer = solution_str
        print("can not extract answer")
    return answer

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def rollback(s: str) -> str:
    python_pos = s.rfind('<python>')
    if python_pos == -1:
        return s

    think_end = '</think>'
    start_pos = python_pos
    if python_pos >= len(think_end) and s[python_pos - len(think_end):python_pos] == think_end:
        start_pos = python_pos - len(think_end)

    pos = start_pos - 1
    while pos >= 0 and (s[pos] == '\n' or s[pos] == ' '):
        pos -= 1

    while pos >= 0:
        if s[pos] == '\n':
            return s[:pos + 1]
        if pos > 0 and s[pos - 1:pos + 1] == '. ':
            return s[:pos + 1]
        pos -= 1
    return s[:pos + 1]

def load_model(config):
    model = LLM(
                config['model_path'],
                dtype=config['type'],
                enforce_eager=True,
                trust_remote_code=True,
                max_model_len=config['max_input_len'],
                gpu_memory_utilization=config['gpu_use'],
                tensor_parallel_size=config['gpu_num'],
            )
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    return model, tokenizer


def extract_multiple_queries(text: str) -> List[str]:
    """
    从文本中提取多个<query></query>标签包围的查询
    """
    queries = []
    pattern = r'<query>(.*?)</query>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        query = match.strip()
        if query:  # 只添加非空查询
            queries.append(query)
    
    # 限制最多3个查询
    return queries[:3]
