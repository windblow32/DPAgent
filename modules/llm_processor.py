import os
import json
from openai import OpenAI

class LLMProcessor:
    def __init__(self):
        """初始化LLM处理器"""
        # 从环境变量获取API密钥和基础URL
        self.api_key = "sk-Sz5VQcsOmLGRz0Ne837cEc158d9f477292B856335cEfD361"
            
        self.api_base = "https://api.gpt.ge/v1/"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.gpt.ge/v1/",
            default_headers={"x-foo": "true"}
        )

    def parse_user_request(self, user_input):
        """解析用户输入，返回处理请求的参数"""
        print(f"Processing user input: {user_input}")
        print(f"Using API key: {self.api_key[:10]}...")
        print(f"Using API base: {self.api_base}")
        
        try:
            # 检查用户输入是否包含"分析"关键词
            if "分析" in user_input:
                return {
                    "operation": "all",
                    "method": "zscore",
                    "columns": [],
                    "explanation": "用户需要完整的数据分析流程"
                }
            
            # 发送请求到OpenAI API
            print("Sending request to OpenAI API...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """你是一个数据处理助手。
                    请解析用户的数据处理需求，并返回JSON格式的处理参数。
                    operation可选值：normalizer, all
                    method可选值：zscore, minmax, decimal, log10
                    columns: 需要处理的列名列表
                    
                    示例输出：
                    {
                        "operation": "normalizer",
                        "method": "zscore",
                        "columns": ["age", "salary"],
                        "explanation": "用户需要对age和salary列进行z-score标准化"
                    }
                    """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            
            print("Received response from OpenAI API")
            content = response.choices[0].message.content
            print(f"Raw response: {content}")
            
            # 解析JSON响应
            result = json.loads(content)
            print(f"Successfully parsed request: {result}")
            return result
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return None

    def process_request(self, prompt: str) -> str:
        """处理通用请求，返回文本响应"""
        print("发送请求到OpenAI API...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据处理助手。请按照用户的要求处理数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            print("收到OpenAI API响应")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"API请求错误：{str(e)}")
            raise
