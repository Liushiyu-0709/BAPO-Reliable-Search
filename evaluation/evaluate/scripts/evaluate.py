from math import fabs
import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time, datetime, sys
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.math_equivalence import is_equiv
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List, final
import time

# add the file's dir into the system path

def extract_answer_fn(output, mode='qa', extract_answer=False):
    if output is None:
        return 'None'
    if extract_answer == False and mode not in ['infogen', 'summary']:
        if mode == 'qa':
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = output
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary']:
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].strip('\n').strip(':**').strip("```").strip()
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])
        else:
            extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])
        extracted_text = extracted_text[:20000]
    elif mode in ['math', 'choose', 'qa']:
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            if '\\boxed' in output:
                idx = output.rfind('\\boxed')
                i = idx
                right_brace_idx = None
                num_left_braces_open = 0
                while i < len(output):
                    if output[i] == '{':
                        num_left_braces_open += 1
                    elif output[i] == '}':
                        num_left_braces_open -= 1
                        if num_left_braces_open == 0:
                            right_brace_idx = i
                            break
                    i += 1

                if right_brace_idx is not None:
                    extracted_text = output[idx + len('\\boxed'):right_brace_idx]
                else:
                    extracted_text = output[idx:]
        else:
            pattern = 'ANSWER:'
            if pattern in output:
                extracted_text = output.split(pattern)[-1].strip('**').strip()
        if '\\text' in extracted_text:
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text


async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
) -> bool:
    """Evaluate a single pair of answers using LLM"""
    prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                reason_answer = chat_response.choices[0].message.content.strip()
                try:
                    start = reason_answer.index("<judgment>") + len("<judgment>")
                    end = reason_answer.index("</judgment>")
                    response_text = reason_answer[start:end].strip()
                except:
                    response_text = reason_answer.strip()
                llm_judge = is_equiv(pred_answer, labeled_answer) or \
                    "correct" in response_text.lower() and \
                    not ("incorrect" in response_text.lower() or \
                         "wrong" in response_text.lower() or \
                         "not correct" in response_text.lower())
                return llm_judge, reason_answer
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                print(f"-------------------pred_answer: {pred_answer}----------------------")
                print(f"-------------------labeled_answer: {labeled_answer}----------------------")
                return is_equiv(pred_answer, labeled_answer), "Error"
            await asyncio.sleep(1 * (attempt + 1))

    return is_equiv(pred_answer, labeled_answer), "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str],
    pred_answers: List[str],
    api_base_url: str = "", #input your api base url
    model_name: str = None, # input model name
    api_key: str = "", # input api key
    concurrent_limit: int = 3,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    semaphore = asyncio.Semaphore(concurrent_limit)

    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result

        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)

    return results

def count_tag_pairs(s, start_tag, end_tag):
    count = 0
    stack = []

    i = 0
    while i < len(s):
        if i <= len(s) - len(start_tag) and s[i:i+len(start_tag)] == start_tag:
            stack.append(i)
            i += len(start_tag)
        elif i <= len(s) - len(end_tag) and s[i:i+len(end_tag)] == end_tag:
            if stack:
                stack.pop()
                count += 1
            i += len(end_tag)
        else:
            i += 1

    return count

def evaluate_predictions(output, labeled_answer, mode='math', use_llm=False, question=None, extract_answer=False, item=None):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'llm_equal': 0}
    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)
    pred_answer_new = pred_answer

    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    else:
        # If no answer was extracted, keep only the last 3 lines
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    tool_counts = 0

    if mode in ['qa']:
        def normalize_answer_qa(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.strip().split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        if not isinstance(pred_answer_new, str):
            pred_answer_new = str(pred_answer_new)
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)
        if type(labeled_answer)==str:
            if labeled_answer[0] !="[":
                labeled_answer = [labeled_answer]
                # print(f"check: {labeled_answer}")
            else:

                import ast
                cleaned = labeled_answer.strip('[]')
                labeled_answer = [e.strip() for e in re.split(r",\s*", cleaned)]
                # print(f"check: {labeled_answer}")

        for answer in labeled_answer:
            if not isinstance(answer, str):
                answer = str(answer)
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        def normalize_answer(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            text = " ".join(text.strip().split())
            return text
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)
        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)

        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
    sequence = item['Full_output']

    try:
        last_answer_end_pos = sequence.rfind("</answer>")
        assert last_answer_end_pos != -1
        sequence = sequence[:last_answer_end_pos + len("</answer>")]
    except:
        try:
            last_answer_end_pos = sequence.rfind("\\boxed")
            assert last_answer_end_pos != -1
            sequence = sequence[:last_answer_end_pos + len("\\boxed")]
        except:
            pass

    if '<python>' in sequence:
        code_counts = count_tag_pairs(sequence, '<python>', '</python>')
        tool_counts += code_counts
    if '<search>' in sequence:
        search_counts = count_tag_pairs(sequence, '<search>', '</search>')
        tool_counts += search_counts
    if '```python' in sequence:
        code_counts = count_tag_pairs(sequence, '```python', '```')
        tool_counts += code_counts
    if '<|begin_of_query|>' in sequence:
        search_counts = count_tag_pairs(sequence, '<|begin_of_query|>', '<|end_of_query|>')
        tool_counts += search_counts

    final_metric["em"] = em
    final_metric["acc"] = acc
    final_metric["f1"] = f1

    final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

        # Add LLM-based evaluation if requested
    if use_llm and question is not None:
        final_metric["llm_equal"] = 0  # Will be updated in batch later
        final_metric["llm_response"] = "LLM evaluation not performed yet"
    final_metric["tool_counts"] = tool_counts
    if tool_counts >= 1:
        final_metric["tool_use"] = 1
    else:
        final_metric["tool_use"] = 0

    return final_metric, pred_answer


def run_evaluation(filtered_data, input_list, output_list, task_type, output_dir, output_metrics_path, output_metrics_overall_path, use_llm=False, extract_answer=False, dataset_name=None, sigma=0.1):
    # Evaluation for math/qa tasks
    avg_em, avg_acc, avg_f1, avg_math, avg_llm, avg_tool, avg_tool_used, avg_passk = [], [], [], [], [], [], [], []

    avg_em_refuse, avg_acc_refuse, avg_f1_refuse, avg_llm_refuse, avg_tool_refuse, avg_passk_refuse, avg_refuse = [], [], [], [], [], [], 0
    num_valid_answer = 0

    # Lists to store data for batch LLM evaluation
    questions_for_llm = []
    labeled_answers_for_llm = []
    pred_answers_for_llm = []
    items_for_llm = []
    items = []

    for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list), total=len(input_list)):
        labeled_answer = item.get('answer', '')  # Use get() to safely access the answer field

        metric, pred_answer = evaluate_predictions(
            output=result,
            labeled_answer=labeled_answer,
            mode=task_type,
            use_llm=use_llm,
            question=input_prompt,
            extract_answer=extract_answer,
            item=item
        )




        item['Pred_Answer'] = pred_answer
        item['Metrics'] = metric
        item['Question'] = input_prompt
        items.append(item)
        # Store data for batch LLM evaluation
        if use_llm:
            questions_for_llm.append(input_prompt)
            labeled_answers_for_llm.append(labeled_answer)
            pred_answers_for_llm.append(pred_answer)
            items_for_llm.append(item)


        # Determine the validity of the predicted answer
        my_method_valid = (pred_answer != '')
        # if passk_key:
            # avg_passk.append(metric[passk_key])
        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])
        avg_tool.append(metric['tool_counts'])
        if metric['tool_use'] >= 1:
            avg_tool_used.append(1)
        else:
            avg_tool_used.append(0)

        if my_method_valid:
            num_valid_answer += 1


        if "I DON'T KNOW".lower() not in pred_answer.lower(): #如果说不是拒绝回答
            avg_refuse += 1
            avg_em_refuse.append(metric['em'])
            avg_acc_refuse.append(metric['acc'])
            avg_f1_refuse.append(metric['f1'])
            avg_tool_refuse.append(metric['tool_counts'])


    # print(avg_em_refuse_judge)


    # Perform batch LLM evaluation if needed
    if use_llm and questions_for_llm:
        llm_results = asyncio.run(llm_evaluate_equivalence_batch(
            questions=questions_for_llm,
            labeled_answers=labeled_answers_for_llm,
            pred_answers=pred_answers_for_llm,
            extract_answer=extract_answer
        ))


        # Update metrics with LLM results
        for item, (llm_result, reason_answer) in zip(items_for_llm, llm_results):
            item['Metrics']['llm_equal'] = int(llm_result)
            item['Metrics']['llm_response'] = reason_answer
            avg_llm.append(int(llm_result))
        for i in range(len(pred_answers_for_llm)):
            if "I DON'T KNOW".lower() not in pred_answers_for_llm[i].lower():
                avg_llm_refuse.append(int(llm_results[i][0]))

    full_outputs = []
    for item in items:
        if item['Output'] != None:
            full_outputs.append(item['Output'])
        else:
            full_outputs.append(item['Full_output'][-200:])

    # Compute overall metrics

    final_tool_productivity = 0.0
    if use_llm:
        final_tool_productivity = np.sum(avg_llm) / np.sum(avg_tool) if np.sum(avg_tool) > 0 else 0.0
    else:
        final_tool_productivity = np.sum(avg_math) / np.sum(avg_tool) if np.sum(avg_tool) > 0 else 0.0

    overall_metrics = {
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0.0,
        'em_reliability': np.mean(avg_em)* (1- (avg_refuse / len(input_list))) + np.mean(avg_em_refuse) * ((avg_refuse / len(input_list))),
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        'tool_call': np.mean(avg_tool) if len(avg_tool) > 0 else 0.0,
        'tool_productivity': final_tool_productivity,
        'average_datas_used_tool_number': np.mean(avg_tool_used) if len(avg_tool_used) > 0 else 0.0,
        'em_REFUSE': np.mean(avg_em_refuse) if len(avg_em_refuse) > 0 else 0.0,
        'f1_REFUSE': np.mean(avg_f1_refuse) if len(avg_f1_refuse) > 0 else 0.0,
        'llm_REFUSE': np.mean(avg_llm_refuse) if len(avg_llm_refuse) > 0 else 0.0,
        'llm_REFUSE': np.mean(avg_llm) if len(avg_llm) > 0 else 0.0,
        'llm_reliability': np.mean(avg_llm)* (1- (avg_refuse / len(input_list))) + np.mean(avg_llm_refuse) * ((avg_refuse / len(input_list))),
        'tool_call_REFUSE': np.mean(avg_tool_refuse) if len(avg_tool_refuse) > 0 else 0.0,
        'Answer_RATE': avg_refuse / len(input_list),
    }

    # Add LLM evaluation metric if available
    if len(avg_llm) > 0:
        overall_metrics['llm_equal'] = np.mean(avg_llm)
        overall_metrics['m1m2'] = overall_metrics['average_datas_used_tool_number'] * overall_metrics['llm_equal'] / (overall_metrics['tool_call'] + sigma) * 100.0

    mytime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_metrics['datetime'] = mytime
    overall_metrics['dataset_name'] = dataset_name

    print(json.dumps(overall_metrics, indent=4))

    # Save prediction results and metrics
    with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(output_metrics_overall_path, mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--task', type=str, required=True, choices=['code', 'math', 'choose', 'qa', 'llm'], help='Task type for evaluation')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM for equivalence evaluation')
    parser.add_argument('--extract_answer', action='store_true', help='Extract answer from output')
    parser.add_argument('--api_base_url', type=str, default=None, help='Base URL for LLM API')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for LLM evaluation')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name for LLM evaluation')
    args = parser.parse_args()

    output_path = args.output_path
    output_metrics_path = output_path.replace('.json', '.metrics.json')
    output_metrics_overall_path = output_path.replace('.json', '.metrics.overall.json')

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare input_list and output_list for run_evaluation
    input_list = []
    output_list = []
    filtered_data = []

    if isinstance(data, dict):
        # Convert dict to list if data is a dictionary
        for key, item in data.items():
            if isinstance(item, dict):  # Ensure item is a dictionary
                filtered_data.append(item)
                input_list.append(item.get('question'))
                output_list.append(item.get('result'))
    else:
        filtered_data = data
        input_list = []
        for item in data:
            prompt = item['Prompt']
            start_index = prompt.find("<|im_start|>user") + len("<|im_start|>user")
            end_index = prompt.find("<|im_end|>\n<|im_start|>assistant")
            question = prompt[start_index:end_index].strip()
            input_list.append(question)

        output_list = []
        count = 0
        for item in data:

            output = item.get('Output', '')
            if output == None:
                output = item.get("Full_output", '')
                lines = output.split('\n')
                output = '\n'.join(lines[-10:])
            if len(output) > 200:
                count+=1
                lines = output.split('\n')
                output = '\n'.join(lines[-10:])

            output_list.append(output)
    # Run evaluation with domain fields
    print(f"There are {count} datas for more than 200 tokens")
    print('------------------------------------------')
    run_evaluation(
        filtered_data=filtered_data,  # Pass the properly structured data
        input_list=input_list,
        output_list=output_list,
        task_type=args.task,
        output_dir=output_path,
        output_metrics_path=output_metrics_path,
        output_metrics_overall_path=output_metrics_overall_path,
        use_llm=args.use_llm,
        extract_answer=args.extract_answer,
        dataset_name=args.dataset_name
    )

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
    print('------------------------------------------')