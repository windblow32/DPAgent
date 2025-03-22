from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from modules.normalizer import DataNormalizer
from modules.transformer import DataTransformer
from modules.entity_matcher import EntityMatcher
from modules.feature_selector import FeatureSelector
from modules.llm_processor import LLMProcessor
from modules.visualizer import DataVisualizer
import json
import logging

app = Flask(__name__)

# Initialize our data preparation components
normalizer = DataNormalizer()
transformer = DataTransformer()
matcher = EntityMatcher()
feature_selector = FeatureSelector()
llm_processor = LLMProcessor()
visualizer = DataVisualizer()

# 全局变量用于存储数据
normalized_result = None

# HTML template for visualization
VIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Normalization Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-title {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }
        .data-container {
            margin-top: 20px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Normalization Results</h1>
        
        <div class="plot-container">
            <div class="plot-title">Distribution Comparison</div>
            <img src="data:image/png;base64,{{ plots.distribution_plot }}" alt="Distribution Plot">
        </div>
        
        <div class="plot-container">
            <div class="plot-title">Box Plot Comparison</div>
            <img src="data:image/png;base64,{{ plots.box_plot }}" alt="Box Plot">
        </div>
        
        <div class="data-container">
            <h2>Original Data (First 5 rows)</h2>
            {{ original_data | safe }}
            
            <h2>Normalized Data (First 5 rows)</h2>
            {{ normalized_data | safe }}
        </div>
    </div>
</body>
</html>
"""

# HTML template for the main page
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>智能数据处理助手</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            min-height: 60px;
        }
        input[type="file"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .success {
            color: #28a745;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #28a745;
            border-radius: 4px;
            background-color: #d4edda;
            display: none;
        }
        .error {
            color: #dc3545;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #dc3545;
            border-radius: 4px;
            background-color: #f8d7da;
            display: none;
        }
        #uploadStatus {
            margin-top: 10px;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-title {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        #result {
            margin-top: 30px;
            display: none;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loading:after {
            content: '处理中...';
            animation: dots 1.5s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: '处理中'; }
            40% { content: '处理中.'; }
            60% { content: '处理中..'; }
            80% { content: '处理中...'; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>智能数据处理助手</h1>
        
        <div class="input-section">
            <form id="processingForm">
                <div class="form-group">
                    <label for="requirement">请描述您的数据处理需求：</label>
                    <textarea id="requirement" name="requirement" placeholder="例如：'我需要对数据进行标准化处理'或'请帮我找出两个表中相似的记录'"></textarea>
                </div>
                
                <div class="form-group">
                    <label for="data">上传数据文件：</label>
                    <input type="file" id="data" name="data" class="file-input" accept=".csv,.json,.xlsx" required>
                    <div id="uploadStatus1" class="success"></div>
                </div>
                
                <div class="form-group" id="data2Container" style="display: none;">
                    <label for="data2">上传第二个数据文件（用于实体匹配）：</label>
                    <input type="file" id="data2" name="data2" class="file-input" accept=".csv,.json,.xlsx">
                    <div id="uploadStatus2" class="success"></div>
                </div>
                
                <button type="submit">处理数据</button>
            </form>
        </div>

        <div class="loading"></div>
        
        <div id="result">
            <div id="errorMessage" class="error"></div>
            
            <div id="matchInfo"></div>
            
            <div id="plotContainer" class="plot-container">
                <h2>可视化结果</h2>
                <div id="plots"></div>
            </div>
            
            <div id="dataContainer" class="table-container">
                <h2>数据详情</h2>
                <div id="data"></div>
            </div>
        </div>
    </div>

    <script>
        async function checkFileUpload(file, statusElement) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload_status', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // 清除之前的状态
                statusElement.classList.remove('success', 'error');
                
                if (result.success) {
                    statusElement.classList.add('success');
                    statusElement.style.display = 'block';
                } else {
                    statusElement.classList.add('error');
                    statusElement.style.display = 'block';
                }
                
                statusElement.textContent = result.message;
                
                // 3秒后隐藏提示
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 3000);
                
            } catch (error) {
                statusElement.classList.remove('success');
                statusElement.classList.add('error');
                statusElement.style.display = 'block';
                statusElement.textContent = '上传出错：' + error.message;
                
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 3000);
            }
        }

        document.getElementById('requirement').addEventListener('input', function(e) {
            const text = e.target.value.toLowerCase();
            const data2Container = document.getElementById('data2Container');
            if (text.includes('两个') && (text.includes('表') || text.includes('文件')) && text.includes('相似')) {
                data2Container.style.display = 'block';
            } else {
                data2Container.style.display = 'none';
            }
        });

        // 监听文件上传
        document.getElementById('data').addEventListener('change', function(e) {
            if (this.files.length > 0) {
                checkFileUpload(this.files[0], document.getElementById('uploadStatus1'));
            }
        });

        document.getElementById('data2').addEventListener('change', function(e) {
            if (this.files.length > 0) {
                checkFileUpload(this.files[0], document.getElementById('uploadStatus2'));
            }
        });

        // 处理表单提交
        document.getElementById('processingForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // 显示加载动画
            document.querySelector('.loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const formData = new FormData(this);
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // 解析返回的HTML
                const parser = new DOMParser();
                const doc = parser.parseFromString(result.html, 'text/html');
                
                // 检查是否有错误
                const error = doc.querySelector('.error');
                if (error) {
                    document.getElementById('errorMessage').textContent = error.textContent;
                    document.getElementById('errorMessage').style.display = 'block';
                    document.getElementById('plotContainer').style.display = 'none';
                    document.getElementById('dataContainer').style.display = 'none';
                } else {
                    // 更新匹配信息
                    const matchInfo = doc.querySelector('.match-info');
                    if (matchInfo) {
                        document.getElementById('matchInfo').innerHTML = matchInfo.innerHTML;
                    }
                    
                    // 更新图表
                    const plots = doc.querySelectorAll('.plot-container img');
                    if (plots.length > 0) {
                        document.getElementById('plotContainer').style.display = 'block';
                        document.getElementById('plots').innerHTML = '';
                        plots.forEach(plot => {
                            document.getElementById('plots').appendChild(plot.cloneNode(true));
                        });
                    } else {
                        document.getElementById('plotContainer').style.display = 'none';
                    }
                    
                    // 更新数据表格
                    const dataTable = doc.querySelector('.table-container table');
                    if (dataTable) {
                        document.getElementById('dataContainer').style.display = 'block';
                        document.getElementById('data').innerHTML = dataTable.outerHTML;
                    } else {
                        document.getElementById('dataContainer').style.display = 'none';
                    }
                    
                    document.getElementById('errorMessage').style.display = 'none';
                }
                
                // 显示结果区域
                document.getElementById('result').style.display = 'block';
                
            } catch (error) {
                document.getElementById('errorMessage').textContent = '处理出错: ' + error.message;
                document.getElementById('errorMessage').style.display = 'block';
                document.getElementById('plotContainer').style.display = 'none';
                document.getElementById('dataContainer').style.display = 'none';
                document.getElementById('result').style.display = 'block';
            }
            
            // 隐藏加载动画
            document.querySelector('.loading').style.display = 'none';
        });
    </script>
</body>
</html>
"""

# HTML template for result page
RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>数据处理结果</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .table { margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div id="results">
            {% if error %}
                <div class="alert alert-danger">{{ error }}</div>
            {% else %}
                {{ content | safe }}
            {% endif %}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(MAIN_TEMPLATE)

@app.route('/upload_status', methods=['POST'])
def upload_status():
    """处理文件上传状态检查"""
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'message': '没有选择文件'
            })

        # 检查文件格式
        if not file.filename.endswith(('.csv', '.json', '.xlsx')):
            return jsonify({
                'success': False,
                'message': '不支持的文件格式，请上传 .csv, .json 或 .xlsx 文件'
            })

        # 尝试读取文件以验证其完整性
        try:
            if file.filename.endswith('.csv'):
                pd.read_csv(file)
            elif file.filename.endswith('.json'):
                pd.read_json(file)
            else:
                pd.read_excel(file)
            
            return jsonify({
                'success': True,
                'message': f'成功上传文件：{file.filename}'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'文件格式错误：{str(e)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'上传失败：{str(e)}'
        })

@app.route('/show_normalized_result')
def show_normalized_result():
    if normalized_result is None:
        return "No data available"
    
    original_data, processed_data, used_columns = normalized_result
    
    return f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>标准化结果</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <h2>数据标准化结果</h2>
            <div class="alert alert-info">
                <strong>处理信息：</strong><br>
                - 使用Z-score标准化方法<br>
                - 处理的列：{', '.join(used_columns)}
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h4>原始数据</h4>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        {original_data.to_html(classes='table table-striped table-bordered', index=True)}
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h4>标准化后的数据</h4>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        {processed_data.to_html(classes='table table-striped table-bordered', index=True)}
                    </div>
                </div>
            </div>
            
            <div class="mt-3">
                <a href="/" class="btn btn-primary">返回首页</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/process', methods=['POST'])
def process_request():
    try:
        # 获取用户输入
        requirement = request.form.get('requirement')
        if not requirement:
            return jsonify({
                'success': False,
                'html': '<div class="alert alert-danger">请输入您的数据处理需求</div>'
            })
            
        print(f"Processing requirement: {requirement}")  # Debug log
            
        # 使用LLM解析用户需求
        parsed_request = llm_processor.parse_user_request(requirement)
        if not parsed_request:
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, 
                    error='无法理解您的需求。请尝试以下表述：<br>' + 
                    '- "找出两个表中相似的记录"<br>' + 
                    '- "请对name和address列进行模糊匹配"<br>' + 
                    '- "标准化处理数据"')
            })
        
        print(f"Parsed request: {parsed_request}")  # Debug log
        
        # 获取数据文件
        if 'data' not in request.files:
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, error='请上传数据文件')
            })
            
        file = request.files['data']
        if file.filename == '':
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, error='请选择文件')
            })
            
        print(f"Processing file: {file.filename}")  # Debug log
            
        # 读取数据
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.filename.endswith('.json'):
                data = pd.read_json(file)
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(file)
            else:
                return jsonify({
                    'success': False,
                    'html': render_template_string(RESULT_TEMPLATE, 
                        error='不支持的文件格式。请上传 .csv, .json 或 .xlsx 文件')
                })
                
            print(f"Successfully loaded file with shape: {data.shape}")  # Debug log
            
        except Exception as e:
            print(f"Error loading file: {str(e)}")  # Debug log
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, error=f'读取文件时出错：{str(e)}')
            })
            
        # 获取操作类型和方法
        operation = parsed_request['operation']
        method = parsed_request['method']
        columns = parsed_request.get('columns', [])
        
        print(f"Operation: {operation}, Method: {method}, Columns: {columns}")  # Debug log
        
        # 根据解析结果调用相应的处理函数
        if operation == 'normalizer':
            processed_data, used_columns, plots = normalizer.normalize(data, method=method)
            print("output processed_data", processed_data)
            
            # 存储结果
            global normalized_result
            normalized_result = (data, processed_data, used_columns)
            
            # 返回成功消息和查看结果的链接
            success_html = '''
                <div class="alert alert-success">
                    <h4>数据标准化处理完成！</h4>
                    <p>使用了Z-score标准化方法，处理了以下列：{}</p>
                    <div class="mt-3">
                        <button onclick="window.open('http://127.0.0.1:2366/show_normalized_result', '_blank')" class="btn btn-primary btn-lg">
                            <i class="bi bi-table"></i> 查看标准化结果
                        </button>
                    </div>
                </div>
            '''.format(', '.join(used_columns))
            
            return jsonify({
                'success': True,
                'html': success_html
            })
            
        elif operation == 'transformer':
            processed_data, used_columns, plots = transformer.transform(data, method=method)
            return jsonify({
                'success': True,
                'html': render_template_string(
                    RESULT_TEMPLATE,
                    operation=operation,
                    columns=used_columns,
                    plots=plots,
                    data=processed_data.to_html(classes='table table-striped')
                )
            })
            
        elif operation == 'entity_matcher':
            # 检查用户输入是否包含"两个"表格的比较需求
            needs_two_files = '两个' in requirement and ('表' in requirement or '文件' in requirement) and '相似' in requirement
            
            if needs_two_files:
                # 检查是否上传了第二个文件
                if 'data2' not in request.files:
                    return jsonify({
                        'success': False,
                        'html': render_template_string(RESULT_TEMPLATE, 
                            error='需要上传两个文件进行比较。请上传第二个文件。')
                    })
                    
                file2 = request.files['data2']
                if file2.filename == '':
                    return jsonify({
                        'success': False,
                        'html': render_template_string(RESULT_TEMPLATE, error='请选择第二个文件')
                    })
                    
                # 读取第二个文件
                try:
                    if file2.filename.endswith('.csv'):
                        data2 = pd.read_csv(file2)
                    elif file2.filename.endswith('.json'):
                        data2 = pd.read_json(file2)
                    elif file2.filename.endswith('.xlsx'):
                        data2 = pd.read_excel(file2)
                    else:
                        return jsonify({
                            'success': False,
                            'html': render_template_string(RESULT_TEMPLATE, 
                                error='不支持的文件格式。请上传 .csv, .json 或 .xlsx 文件')
                        })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'html': render_template_string(RESULT_TEMPLATE, error=f'读取第二个文件时出错：{str(e)}')
                    })
            else:
                # 如果不需要两个文件，就在同一个文件内进行匹配
                data2 = data.copy()
            
            try:
                # 使用指定的列进行匹配
                matches = matcher.match(data, data2, columns=columns)
                
                if len(matches) == 0:
                    return jsonify({
                        'success': False,
                        'html': render_template_string(RESULT_TEMPLATE, 
                            error='未找到匹配的记录。请检查数据或尝试其他匹配列。')
                    })
                
                # 生成可视化
                plots = visualizer.visualize_data(matches)
                
                # 添加匹配信息
                match_info = f"""
                <div class="match-info">
                    <h3>匹配结果</h3>
                    <p>找到 {len(matches)} 条匹配的记录</p>
                    <p>匹配列：{', '.join(columns)}</p>
                    <p>结果表格大小：{matches.shape[0]} 行 × {matches.shape[1]} 列</p>
                </div>
                """
                
                return jsonify({
                    'success': True,
                    'html': render_template_string(
                        RESULT_TEMPLATE,
                        operation=operation,
                        columns=columns,
                        plots=plots,
                        match_info=match_info,
                        data=matches.to_html(classes='table table-striped')
                    )
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'html': render_template_string(RESULT_TEMPLATE, error=f'实体匹配过程中出错：{str(e)}')
                })
            
        elif operation == 'data_augmentation':
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, error='数据增强功能正在开发中...')
            })
            
        else:
            return jsonify({
                'success': False,
                'html': render_template_string(RESULT_TEMPLATE, error='不支持的操作类型')
            })
            
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}\n{logging.format_exc()}")
        return jsonify({
            'success': False,
            'html': render_template_string(RESULT_TEMPLATE, error=f'处理出错：{str(e)}')
        })

@app.route('/normalize', methods=['POST'])
def normalize_data():
    """
    Normalize data using specified method.
    Expected JSON input:
    {
        "data": [[1.0, 2.0], [3.0, 4.0]],
        "method": "standard" | "minmax" | "robust",
        "columns": ["col1", "col2"]  // optional
    }
    """
    try:
        content = request.json
        if not content or 'data' not in content:
            return jsonify({'error': 'No data provided'}), 400

        # Get parameters
        data = content['data']
        method = content.get('method', 'standard')
        columns = content.get('columns', None)

        # Convert to DataFrame if columns are provided
        if columns:
            data = pd.DataFrame(data, columns=columns)

        # Normalize data with visualization
        normalized_data, used_columns, plots = normalizer.normalize(data, method, columns)

        # Prepare data for display
        if isinstance(normalized_data, pd.DataFrame):
            original_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            original_html = original_df.head().to_html(classes='table table-striped')
            normalized_html = normalized_data.head().to_html(classes='table table-striped')
        else:
            original_df = pd.DataFrame(data)
            normalized_df = pd.DataFrame(normalized_data)
            original_html = original_df.head().to_html(classes='table table-striped')
            normalized_html = normalized_df.head().to_html(classes='table table-striped')

        # If plots are available, return HTML visualization
        if plots:
            return jsonify({
                'success': True,
                'html': render_template_string(
                    VIZ_TEMPLATE,
                    plots=plots,
                    original_data=original_html,
                    normalized_data=normalized_html
                )
            })
        
        # Otherwise return JSON response
        if isinstance(normalized_data, pd.DataFrame):
            output_data = normalized_data.to_dict(orient='records')
        else:
            output_data = normalized_data.tolist()

        return jsonify({
            'status': 'success',
            'normalized_data': output_data,
            'columns': used_columns if used_columns is not None else [],
            'method': method
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/transform', methods=['POST'])
def transform_data():
    """
    Transform data using specified method.
    Expected JSON input:
    {
        "data": [[1.0, 2.0], [3.0, 4.0]],
        "method": "yeo-johnson" | "box-cox" | "quantile_normal" | "quantile_uniform",
        "columns": ["col1", "col2"]  // optional
    }
    """
    try:
        content = request.json
        if not content or 'data' not in content:
            return jsonify({'error': 'No data provided'}), 400

        # Get parameters
        data = content['data']
        method = content.get('method', 'yeo-johnson')
        columns = content.get('columns', None)

        # Convert to DataFrame if columns are provided
        if columns:
            data = pd.DataFrame(data, columns=columns)

        # Transform data
        transformed_data, used_columns = transformer.transform(data, method, columns)

        # Convert output to list for JSON serialization
        if isinstance(transformed_data, pd.DataFrame):
            output_data = transformed_data.to_dict(orient='records')
        else:
            output_data = transformed_data.tolist()

        return jsonify({
            'status': 'success',
            'transformed_data': output_data,
            'columns': used_columns if used_columns is not None else [],
            'method': method
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/match', methods=['POST'])
def entity_matching():
    """
    Match entities between two datasets.
    Expected JSON input:
    {
        "data1": [{"id": 1, "name": "John"}, ...],
        "data2": [{"id": 1, "name": "Johnny"}, ...],
        "method": "exact" | "fuzzy" | "tfidf",
        "columns": ["name"],  // columns to use for matching
        "threshold": 0.8  // similarity threshold (optional)
    }
    """
    try:
        content = request.json
        if not content or 'data1' not in content or 'data2' not in content:
            return jsonify({'error': 'Both datasets are required'}), 400

        # Get parameters
        data1 = pd.DataFrame(content['data1'])
        data2 = pd.DataFrame(content['data2'])
        # fixme default method should be exact
        method = content.get('method', 'exact')
        columns = content.get('columns')
        threshold = content.get('threshold', 0.8)

        if columns is None:
            return jsonify({'error': 'Matching columns must be specified'}), 400

        # Perform matching
        matches = matcher.match(data1, data2, method, columns, threshold)

        return jsonify({
            'status': 'success',
            'matches': matches.to_dict(orient='records'),
            'method': method,
            'threshold': threshold
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/select-features', methods=['POST'])
def feature_selection():
    """
    Select features using specified method.
    Expected JSON input:
    {
        "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...],
        "target": [0, 1, 0, ...],
        "method": "kbest" | "mutual_info" | "random_forest",
        "n_features": 2,  // number of features to select
        "feature_names": ["feature1", "feature2", "feature3"]  // optional
    }
    """
    try:
        content = request.json
        if not content or 'features' not in content or 'target' not in content:
            return jsonify({'error': 'Features and target are required'}), 400

        # Get parameters
        features = content['features']
        target = content['target']
        method = content.get('method', 'kbest')
        n_features = content.get('n_features', 2)
        feature_names = content.get('feature_names')

        # Convert to DataFrame if feature names are provided
        if feature_names:
            features = pd.DataFrame(features, columns=feature_names)

        # Select features
        selected_features, importance_df = feature_selector.select_features(
            features, target, method, n_features
        )

        return jsonify({
            'status': 'success',
            'selected_features': selected_features.tolist(),
            'feature_importance': importance_df.to_dict(orient='records'),
            'method': method
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=2366)
