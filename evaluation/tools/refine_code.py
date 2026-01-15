from urllib.parse import urljoin
import requests
import time 
from argparse import Namespace
import asyncio
from openai import OpenAI
from transformers import AutoTokenizer

def refine(prompt, response):

    API_BASE_URL = "your_api_base_url"
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )

    prompt = f"""You are an expert in response refinement. Given a prompt and its corresponding response, your task is to compress and restructure the response by removing redundant, repetitive, or irrelevant content. Preserve all key information needed to directly and accurately address the original prompt. Only output your revised response and do not output anything else.
    **Original Prompt:**
    {prompt}
    **Original Response:**
    {response}
    **Revised Response:**
    """

    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    
    response_text = chat_response.choices[0].message.content.strip()
    
    return response_text






if __name__ == "__main__":

    # 测试用例：一个有语法错误的代码
    code = """
# 创建符号变量
x, y = symbols('x y')

# 定义方程
eq1 = x**2 + y**2 = 25  # 错误的等号使用
eq2 = 2*x + y = 10

# 求解方程组
solution = solve((eq1, eq2), (x, y))
print(f"Solution: {solution}")
    """
    
    error = "SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '=' ?"
    