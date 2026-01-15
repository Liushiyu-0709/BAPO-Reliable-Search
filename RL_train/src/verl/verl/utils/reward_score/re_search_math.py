# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import signal
import re
from contextlib import contextmanager
import importlib.util
from verl.utils.reward_score.eval import normalize_final_answer
from verl.utils.reward_score.re_search import validate_format, extract_answer
from transformers import AutoTokenizer
# from verl.utils.reward_score.eval import parse, verify
from math_verify import parse, verify
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

timeout_seconds=2
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
english_pattern = re.compile(r'[a-zA-Z]')
boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')

def check_mixed_languages(text):
    chinese_chars = len(chinese_pattern.findall(text))
    english_chars = len(english_pattern.findall(text))
    return chinese_chars >= 20 and english_chars >= 20

def undesired_format(text):
    if "<|endoftext|>" not in text: return True
    else: return False


def check_garbled_characters(text):
    valid_chars = valid_char_pattern.sub('', text)
    if not text: 
        return False
    invalid_ratio = len(valid_chars) / len(text)
    return invalid_ratio > 0.3

def has_repeated_patterns(text):
    return bool(repeat_pattern.search(text))
    
def correctness_score_default(tokenizer, response, gt):
    if "<|im_start|>assistant\n" in response:
        response_split = response.split("<|im_start|>assistant\n")
    else:
        response_split = response.split("Assistant:")
    
    response = response_split[1]
    # import pdb; pdb.set_trace()
    valid_template, reason = validate_format(response)
    if not valid_template:
        print(f"--------------------------------search/python bad format--------------------------------")
        return 0, f'search/python bad format: {reason}'
    

    

    answer_part = extract_answer(response)
    ground_truth = gt
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            print(f"--------------------------------find box error--------------------------------")
            return 0, f'find box error: {e}'
    else:
        print(f"--------------------------------cannot extract answer--------------------------------")
        return 0, f'cannot extract answer'
    
    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        print(f"--------------------------------over length--------------------------------")
        return 0, f'over length'
    
    # if f1_score > 0:
    #     print(f"--------------------------------correct answer--------------------------------")
    #     return f1_score, f'correct answer, get f1 score: {f1_score}'
    # else:
    #     print(f"--------------------------------wrong answer--------------------------------")
    #     return 0.1, f'wrong answer but good format: {answer}'
    


    # matches = boxed_pattern.findall(response)
    # if not matches: return -1.0
    # pred = matches[-1][:-1]
    print(f"answer_part: {answer}, ground_truth: {ground_truth}")
    if is_equiv(answer, ground_truth):
        print(f"--------------------------------correct answer--------------------------------")
        return 1.0, f'correct answer'
    else:
        print(f"--------------------------------wrong answer--------------------------------")
        return 0.1, f'format good but wrong answer'



def correctness_score_v2(response, gt):
    matches = boxed_pattern.findall(response)
    if not matches: return -1.0
    pred = matches[-1][:-1]
    return 1.0 if is_equiv(pred, gt) else -0.5

def compute_score(tokenizer, solution_str, ground_truth, reward_type) -> float:      
    if reward_type=='default':
        try:     
            
            # if undesired_format(solution_str): return -1.0
            return correctness_score_default(tokenizer, solution_str, ground_truth)
                    
        except TimeoutException:
            return -1.0
        except Exception as e:
            return -1.0
    elif reward_type=="v2.wformat":
        try:
            return correctness_score_v2(solution_str, ground_truth)
        except TimeoutException:
            return -1.0
        except Exception as e:
            return -1.0
    else:
        try:     
            # if undesired_format(solution_str): return -1.0
            return correctness_score_default(solution_str, ground_truth)            
        except TimeoutException:
            return -1.0
        except Exception as e:
            return -1.0




# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    if str1.strip().lower() == str2.strip().lower(): return True
    try:
        str1=normalize_final_answer(str1)
        str2=normalize_final_answer(str2)
        str1=parse(str1)
        str2=parse(str2)
        return verify(str1, str2)
    except:
        pass

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1==ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
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
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


if __name__ == "__main__":
    tokenizer=AutoTokenizer.from_pretrained("/fs/archive/share/u2024001021/huggingface_models/Qwen2.5-1.5B-Instruct")
    # response="To determine which digit appears in the 534th place after the decimal point in the decimal representation of $\\frac{5}{13}$, we need to first find the repeating decimal sequence of $\\frac{5}{13}$. \n\nLet's start by calculating the decimal representation of $\\frac{5}{13}$.\n```python\nfrom decimal import Decimal, getcontext\r\n\r\n# Set the precision high enough to see the repeating pattern clearly\r\ngetcontext().prec = 1000\r\n\r\n# Calculate the decimal representation of 5/13\r\ndecimal_rep = Decimal(5) / Decimal(13)\r\nprint(str(decimal_rep))\n```\n```output\n0.3846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846153846\n```\nThe decimal representation of $\\frac{5}{13}$ is $0.\\overline{384615}$. This means the repeating sequence is \"384615\" and it has a length of 6 digits.\n\nTo find the digit in the 534th place after the decimal point, we need to determine the position within the repeating sequence. Since the sequence repeats every 6 digits, we can find the position by calculating the remainder when 534 is divided by 6.\n\nLet's calculate this.\n```python\n# Length of the repeating sequence\r\nrepeating_sequence = \"384615\"\r\nsequence_length = len(repeating_sequence)\r\n\r\n# Find the position within the repeating sequence\r\nposition = (534 - 1) % sequence_length  # -1 because indexing starts from 0\r\n\r\n# Get the digit at that position\r\ndigit_in_534th_place = repeating_sequence[position]\r\nprint(digit_in_534th_place)\n```\n```output\n6\n```\n<answer>The digit in the 534th place after the decimal point in the decimal representation of $\\frac{5}{13}$ is $\\boxed{6}$.</answer> <|endoftext|>"
    # response="<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To understand the problem, we need context on the methods and their effects. We are given a grid where transforming two neighboring chips costs 1 rouble and swapping two chips with exactly four chips between them is free. The objective is to invert the sequence of chips, and our goal is to calculate the minimum number of roubles required. Let’s start by making a hypothesis on how to effectively utilize these operations.</think>\n\n<think> Let's first investigate what happens when we swap two chips with exactly four chips between them. If we swap, it doesn't consume a rouble but only changes their positions indirectly as stated. We need a strategy to use this free swap operation in a way that minimizes our cost.</think>\n\n<search> search query: What is the grid configuration and initial state given in the problem? </search>\n <result>\n\"Laplacian matrix\"\npoints are specified to have a positive value, while the rest of the values in the grid are zero. Over time, the exponential decay acts to distribute the values at these points evenly throughout the entire grid. The complete Matlab source code that was used to generate this animation is provided below. It shows the process of specifying initial conditions, projecting these initial conditions onto the eigenvalues of the Laplacian Matrix, and simulating the exponential decay of these projected initial conditions. The graph Laplacian matrix can be further viewed as a matrix form of an approximation to the (positive semi-definite)\n\n\"Solution algorithms for pressure-velocity coupling in steady flows\"\ntake the form: The summation covers all nodes and volumes in the immediate vicinity of the selected node. Their values are shown in the next figure. Following this, The algorithms can be applied to get the solutions for the basic staggered grid equations. The acronym stands for Semi- Implicit method for pressure-linked equations. It is essentially a guess and correct procedure for the calculation of the staggered grid pressure field. These have been illustrated using a two dimensional steady flow. Step 1 :- A pressure field p* is guessed. Step 2 :- The Discretized momentum equations are solved using the\n\n\"Riemann problem\"\nof the grid. For that it is widely used in computational fluid dynamics and in MHD simulations. In these fields Riemann problems are calculated using Riemann solvers. As a simple example, we investigate the properties of the one-dimensional Riemann problem in gas dynamics, with initial conditions given by where \"\"x\"\" = 0 separates two different states, together with the linearised gas dynamic equations (see gas dynamics for derivation). where we can assume without loss of generality formula_3. We can now rewrite the above equations in a conservative form: where and the index denotes the partial derivative with respect to the\n\n\"Bayesian model of computational anatomy\"\nComputational anatomy solves for the flow of the Euler-Lagrange equation. Solving the geodesic from the initial condition formula_29 is termed the Riemannian-exponential, a mapping formula_31 at identity to the group. The Riemannian exponential satisfies formula_32 for initial condition formula_33, vector field dynamics formula_34, It is extended to the entire group, formula_41 Depicted in the accompanying figure is a depiction of the random orbits around each exemplar, formula_42, generated by randomizing the flow by generating the initial tangent space vector field at the identity formula_43, and then generating random object formula_44. Shown in the Figure on the right the cartoon orbit,\n\n\"Principles of grid generation\"\ninserted. Logic for establishing new connectivity is determined once the points are inserted. Data that form grid point that identifies grid cell are needed. As each cell is formed it is numbered and the points are sorted. In addition the neighbor cell information is needed. A problem in solving partial differential equations using previous methods is that the grid is constructed and the points are distributed in the physical domain before details of the solution is known. So the grid may or may not be the best for the given problem. Adaptive methods are used to improve the accuracy of\n</result>  \n<think> Upon reviewing the search result, relevant details about grid configurations and initial conditions are not provided, focusing instead on mathematics and numerical solutions. This shows necessity to further clarify or consider other parts of the problem's initial description under which we have operations with chips.\n\nLet's consider the chips represented in a linear sequence of cells rather than a grid, as we are working with neighboring operations for chip swaps. We hypothesize that we use the free swap operation to effectively reverse the sequence without incurrence of roubles, then apply the paying operation in the most strategic way to complete the reversal process if necessary.</think>\n\n<think> Now, let's focus on the operations and the strategy. Here, a fixed length of 100 seems irrelevant to our greedy focus on minimizing the number of operations – it's the movement and interchanging that are important. We've observed that pairwise swaps ranked by exchanging two neighboring chips can enumerate costs for reversibility</think>\n\n<think>We hypothesize that to reverse a sequence, we could swap in pairs over the sequence alternately without incurring costs and only using paying swaps in 'buffered' spaces. Given N chips, we find how many operations are required—and the pattern of sequence reversal hints that the solution is closer to one latency trade-off</think> \n\n<think> Thus with the understanding that we can reverse \\(n\\) items with no cost by optimally leveraging the free swaps, the remaining work involves chip rearrangement by minimally incurring roubles. For two neighboring chips in reverse, swapping results in expenditure per two items with n iterations, totaling cost</think>\n\n<think> Revisiting the problem with a direct calculation, each full reversal across our linear model requires \\(N-1\\) operations. If degeneracy costs \\(1\\) rouble per such processes where \\(N = 100\\). Calculations now clearly link the formula in a temporary swapping with neighbors for cost disposal.</think>\n\n<python>\nN = 100 \ncost_per_pair = 1 \ntotal_poss='N - 1' # As we initially need swaps to buffer the other\ncost = cost_per_pair * (N - 1)\n</python>\n\n <result>\n\n</result>  \n<think> The deducted minimal number of roubles required to do the swap in the reverse linear sequence using optimal strategy aligns calculations as such, which confirms the individual steps in parity.</think>  \n\n<answer> \\[ \\boxed{99} \\] </answer><|im_end|>"
    response="<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To solve the given problem, we will use both natural language reasoning and Python code. Let's break down the problem step-by-step.\n\n1. **Given Equations:**\n   \\[\n   \\log_{10}(\\sin x) + \\log_{10}(\\cos x) = -1\n   \\]\n   Using the properties of logarithms, we can combine the logs:\n   \\[\n   \\log_{10}(\\sin x \\cos x) = -1 \\implies \\sin x \\cos x = 10^{-1} = \\frac{1}{10}\n   \\]\n   Recall the double-angle identity for sine: \\(\\sin(2x) = 2 \\sin x \\cos x\\). So,\n   \\[\n   \\sin(2x) = 2 \\left(\\frac{1}{10}\\right) = \\frac{1}{5}\n   \\]\n\n2. **Second Given Equation:**\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}(\\log_{10}(n) - 1)\n   \\]\n   We can rewrite the right-hand side using properties of logarithms:\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}\\log_{10}(n) - \\frac{1}{2} = \\log_{10}(n^{1\/2}) - \\log_{10}(10^{1\/2}) = \\log_{10}\\left(\\frac{\\sqrt{n}}{\\sqrt{10}}\\right)\n   \\]\n   This implies:\n   \\[\n   \\sin x + \\cos x = \\frac{\\sqrt{n}}{\\sqrt{10}}\n   \\]\n   Squaring both sides, we get:\n   \\[\n   (\\sin x + \\cos x)^2 = \\frac{n}{10}\n   \\]\n   Expanding the left side:\n   \\[\n   \\sin^2 x + \\cos^2 x + 2 \\sin x \\cos x = \\frac{n}{10}\n   \\]\n   Using the Pythagorean identity \\(\\sin^2 x + \\cos^2 x = 1\\) and the earlier result \\(\\sin x \\cos x = \\frac{1}{10}\\):\n   \\[\n   1 + 2 \\left(\\frac{1}{10}\\right) = \\frac{n}{10} \\implies 1 + \\frac{1}{5} = \\frac{n}{10} \\implies \\frac{6}{5} = \\frac{n}{10} \\implies n = 12\n   \\]\n\nNow, let's verify this with Python code to ensure the result is accurate.\n </think><python>\nimport math\r\nfrom sympy import symbols, Eq, solve, log, sin, cos\r\n\r\n# Define the variable\r\nx = symbols('x')\r\n\r\n# First equation: log10(sin(x)) + log10(cos(x)) = -1\r\neq1 = Eq(log(sin(x), 10) + log(cos(x), 10), -1)\r\n\r\n# Solve for sin(x) * cos(x)\r\nsin_x_cos_x = solve(eq1, sin(x) * cos(x))[0]\r\nsin_x_cos_x = sin_x_cos_x.simplify()\r\n\r\n# We know sin(x) * cos(x) = 1\/10\r\nsin_x_cos_x_value = 1 \/ 10\r\n\r\n# Second equation: log10(sin(x) + cos(x)) = 1\/2 * (log10(n) - 1)\r\nn = symbols('n')\r\neq2 = Eq(log(sin(x) + cos(x), 10), (1\/2) * (log(n, 10) - 1))\r\n\r\n# We know (sin(x) + cos(x))^2 = 1 + 2 * sin(x) * cos(x) = 1 + 2 * (1\/10) = 6\/5\r\nsin_x_plus_cos_x_squared = 6 \/ 5\r\nsin_x_plus_cos_x = math.sqrt(sin_x_plus_cos_x_squared)\r\n\r\n# Substitute sin(x) + cos(x) into the second equation\r\nlhs = log(sin_x_plus_cos_x, 10)\r\nrhs = (1\/2) * (log(n, 10) - 1)\r\n\r\n# Solve for n\r\nn_value = solve(Eq(lhs, rhs), n)[0]\r\nprint(n_value)\n</python><result>\n\n12.0000000000000\n</result> <answer>The value of \\( n \\) is \\(\\boxed{12}\\). The calculations and the verification using Python both confirm that the final answer is indeed \\( n = 12 \\).</answer><|im_end|>"
    

    
    answer="12"
    res=compute_score(tokenizer,response, answer, reward_type="default")
    print(res)