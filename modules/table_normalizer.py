import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import re
import traceback
from .llm_processor import LLMProcessor

class TableNormalizer:
    def __init__(self, llm_processor: Optional[LLMProcessor] = None):
        """
        初始化表格规范化器
        
        Args:
            llm_processor: LLM处理器实例，用于生成规范化建议
        """
        self.llm_processor = llm_processor or LLMProcessor()
        
    def _create_normalization_prompt(self, df: pd.DataFrame, title: str = "Unknown") -> str:
        """
        创建用于LLM的规范化提示
        
        Args:
            df: 输入的DataFrame
            title: 表格标题
            
        Returns:
            str: 完整的提示文本
        """
        # 将DataFrame转换为字符串表示
        table_str = df.to_string()
        
        prompt = f"""You are an advanced AI capable of analyzing and understanding information within tables.
Your task is to normalize a web table and convert it into a relational database table, enabling the execution of SQL queries on the data.

### Table:
{table_str}

### Task: 
Your task is to normalize the structure and the values of each cell to convert this table into a regular normalized relational database table.

### Instructions:
1. Extract embedded information into new columns (e.g., from 'John Smith (USA)', extract 'USA' into a new 'Country' column)
2. Normalize dates to YYYY-MM-DD format
3. Clean numerical values:
   - Remove currency symbols ($, €, etc.)
   - Remove thousand separators (commas)
   - Remove units (pts, kg, etc.)
   - Convert percentages to decimal numbers
4. Replace 'N/A', '-', or null values with empty strings
5. Split date ranges (e.g., '2010-2023', '2019/20') into separate start and end columns
6. Preserve all original information, never delete columns or rows
7. Remove any unnecessary special characters

### Output Format:
Return ONLY a JSON array of arrays representing the normalized table, like this:
normalized_table = [
    ["Column1", "Column2", "Column3"],
    ["row1_val1", "row1_val2", "row1_val3"],
    ["row2_val1", "row2_val2", "row2_val3"]
]

Do not include any explanations or other text, just the normalized_table assignment with the JSON array.

### Response:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[List[str]]:
        """
        解析LLM返回的规范化表格结果
        
        Args:
            response: LLM返回的响应文本
            
        Returns:
            List[List[str]]: 解析后的表格数据
        """
        try:
            # 打印原始响应以便调试
            print("\nLLM原始响应:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # 提取normalized_table部分
            start_idx = response.find("normalized_table = ")
            if start_idx == -1:
                start_idx = response.find("normalized_table=")  # 尝试无空格版本
                if start_idx == -1:
                    print("未找到'normalized_table'标记")
                    # 尝试直接解析整个响应
                    table_str = response.strip()
                else:
                    table_str = response[start_idx:].split("=", 1)[1].strip()
            else:
                table_str = response[start_idx:].split("=", 1)[1].strip()
            
            print("\n提取的表格字符串:")
            print("-" * 50)
            print(table_str)
            print("-" * 50)
            
            # 清理和规范化JSON字符串
            table_str = table_str.replace("'", '"')  # 将单引号替换为双引号
            table_str = table_str.replace('\n', '')  # 移除换行符
            table_str = table_str.replace('\r', '')  # 移除回车符
            table_str = table_str.strip()  # 移除首尾空白
            
            # 确保是有效的JSON数组
            if not (table_str.startswith('[') and table_str.endswith(']')):
                print("响应不是有效的JSON数组格式")
                # 尝试提取方括号内的内容
                start = table_str.find('[')
                end = table_str.rfind(']')
                if start != -1 and end != -1:
                    table_str = table_str[start:end+1]
                else:
                    raise ValueError("无法找到有效的JSON数组")
            
            print("\n清理后的JSON字符串:")
            print("-" * 50)
            print(table_str)
            print("-" * 50)
            
            # 使用json.loads解析列表
            try:
                table_data = json.loads(table_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                # 尝试进一步清理
                table_str = table_str.replace(',]', ']')  # 移除尾随逗号
                table_str = re.sub(r',(\s*])', r'\1', table_str)  # 更智能地移除尾随逗号
                table_str = re.sub(r'[\s\t]+', ' ', table_str)  # 规范化空白字符
                print("\n进一步清理后的JSON字符串:")
                print("-" * 50)
                print(table_str)
                print("-" * 50)
                table_data = json.loads(table_str)
            
            if not isinstance(table_data, list):
                raise ValueError("解析后的数据不是列表")
            
            if not all(isinstance(row, list) for row in table_data):
                raise ValueError("解析后的数据不是二维数组")
            
            if not table_data:
                raise ValueError("解析后的数据为空")
            
            print("\n成功解析的数据结构:")
            print(f"行数: {len(table_data)}")
            print(f"列数: {len(table_data[0]) if table_data else 0}")
            
            return table_data
            
        except Exception as e:
            error_msg = f"""
LLM响应解析错误:
原始响应:
{response}

错误信息:
{str(e)}

错误类型:
{type(e).__name__}

堆栈跟踪:
{traceback.format_exc()}
"""
            print(error_msg)
            raise ValueError(f"Failed to parse LLM response: {str(e)}")
    
    def _convert_to_dataframe(self, table_data: List[List[str]]) -> pd.DataFrame:
        """
        将解析后的表格数据转换为DataFrame
        
        Args:
            table_data: 解析后的表格数据
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        if not table_data or len(table_data) < 2:
            raise ValueError("Invalid table data: must contain headers and at least one row")
        
        headers = table_data[0]
        data = table_data[1:]
        
        return pd.DataFrame(data, columns=headers)
    
    def normalize_table(self, df: pd.DataFrame, title: str = "Unknown") -> Tuple[pd.DataFrame, Dict]:
        """
        使用LLM规范化表格
        
        Args:
            df: 输入的DataFrame
            title: 表格标题
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 规范化后的DataFrame和规范化信息
        """
        # 生成提示
        prompt = self._create_normalization_prompt(df, title)
        
        # 获取LLM响应
        response = self.llm_processor.process_request(prompt)
        
        # 解析响应
        table_data = self._parse_llm_response(response)
        
        # 转换为DataFrame
        normalized_df = self._convert_to_dataframe(table_data)
        
        # 收集规范化信息
        info = {
            'original_columns': list(df.columns),
            'normalized_columns': list(normalized_df.columns),
            'original_shape': df.shape,
            'normalized_shape': normalized_df.shape,
            'changes': {
                'new_columns': [col for col in normalized_df.columns if col not in df.columns],
                'modified_columns': [col for col in df.columns if col in normalized_df.columns and not df[col].equals(normalized_df[col])]
            }
        }
        
        return normalized_df, info
    
    def generate_report(self, info: Dict) -> str:
        """
        生成规范化报告
        
        Args:
            info: 规范化信息字典
            
        Returns:
            str: 格式化的报告文本
        """
        report = []
        report.append("表格规范化报告")
        report.append("=" * 30)
        
        # 基本信息
        report.append(f"\n原始表格大小: {info['original_shape']}")
        report.append(f"规范化后表格大小: {info['normalized_shape']}")
        
        # 新增列
        if info['changes']['new_columns']:
            report.append("\n新增列:")
            for col in info['changes']['new_columns']:
                report.append(f"- {col}")
        
        # 修改的列
        if info['changes']['modified_columns']:
            report.append("\n修改的列:")
            for col in info['changes']['modified_columns']:
                report.append(f"- {col}")
        
        return "\n".join(report)
