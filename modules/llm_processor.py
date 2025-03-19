import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

class LLMProcessor:
    def __init__(self):
        self.available_operations = {
            'normalizer': ['standard', 'minmax', 'robust'],
            'transformer': ['yeo-johnson', 'box-cox', 'quantile_normal', 'quantile_uniform'],
            'entity_matcher': ['exact', 'fuzzy', 'tfidf'],
            'data_augmentation': ['noise', 'sampling', 'synthetic']
        }
        
    def parse_user_request(self, user_input):
        print(f"Processing user input: {user_input}")  # Debug log
        
        prompt = f"""作为一个数据处理专家，请分析用户的需求并将其映射到以下可用的数据处理操作中：
        
        可用操作：
        1. normalizer (标准化): standard, minmax, robust
        2. transformer (转换): yeo-johnson, box-cox, quantile_normal, quantile_uniform
        3. entity_matcher (实体匹配): exact, fuzzy, tfidf
        4. data_augmentation (数据增强): 待实现
        
        用户需求: {user_input}
        
        请以JSON格式返回解析结果，格式如下：
        {{
            "operation": "操作类型",
            "method": "具体方法",
            "columns": ["列名1", "列名2"],  # 如果用户指定了要处理的列，返回列名列表；如果没有指定，返回空列表
            "explanation": "为什么选择这个操作和方法"
        }}
        
        示例：
        1. 用户输入："请对name和address列进行模糊匹配"
        返回：{{"operation": "entity_matcher", "method": "fuzzy", "columns": ["name", "address"], "explanation": "用户要求对特定列进行模糊匹配"}}
        
        2. 用户输入："找出两个表中相似的记录"
        返回：{{"operation": "entity_matcher", "method": "fuzzy", "columns": [], "explanation": "用户需要找出相似记录，使用模糊匹配"}}
        
        3. 用户输入："标准化处理数据"
        返回：{{"operation": "normalizer", "method": "standard", "columns": [], "explanation": "用户需要标准化数据"}}
        
        只返回JSON，不要其他解释。"""

        try:
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE')
            
            if not api_key:
                print("Error: OPENAI_API_KEY not found in environment variables")
                return None
                
            if not api_base:
                print("Error: OPENAI_API_BASE not found in environment variables")
                return None
                
            print(f"Using API key: {api_key[:8]}...")  # Only print first 8 characters for security
            print(f"Using API base: {api_base}")
            
            client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            
            print("Sending request to OpenAI API...")  # Debug log
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据处理专家，负责解析用户的数据处理需求。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            print("Received response from OpenAI API")  # Debug log
            print(f"Raw response: {response.choices[0].message.content}")  # Debug log
            
            result = json.loads(response.choices[0].message.content)
            
            # 验证返回的操作和方法是否在可用列表中
            if result['operation'] not in self.available_operations:
                print(f"Invalid operation: {result['operation']}")  # Debug log
                return None
                
            if result['method'] not in self.available_operations[result['operation']]:
                print(f"Invalid method: {result['method']} for operation {result['operation']}")  # Debug log
                return None
                
            print(f"Successfully parsed request: {result}")  # Debug log
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")  # Debug log
            print(f"Raw response: {response.choices[0].message.content}")  # Debug log
            return None
        except Exception as e:
            print(f"Error in parse_user_request: {str(e)}")  # Debug log
            print(f"Error type: {type(e).__name__}")  # Debug log
            return None
