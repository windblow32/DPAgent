document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const dataPreview = document.getElementById('dataPreview');
    const processBtn = document.getElementById('processBtn');
    const userInput = document.getElementById('userInput');
    const resultsContent = document.getElementById('resultsContent');
    const results = document.getElementById('results');

    let uploadedFile = null;
    let secondFileInput = null;

    // 处理文件上传
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('file');
        uploadedFile = fileInput.files[0];
        if (uploadedFile) {
            dataPreview.style.display = 'block';
        }
    });

    // 处理数据处理请求
    processBtn.addEventListener('click', async function() {
        if (!uploadedFile) {
            alert('请先上传数据文件');
            return;
        }

        const userText = userInput.value.trim();
        if (!userText) {
            alert('请输入处理需求');
            return;
        }

        // 显示加载提示
        results.style.display = 'block';
        resultsContent.innerHTML = '<div class="alert alert-info">正在处理数据，请稍候...</div>';

        const formData = new FormData();
        formData.append('requirement', userText);
        formData.append('data', uploadedFile);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                // 显示处理结果
                resultsContent.innerHTML = result.html;
                // 滚动到结果区域
                results.scrollIntoView({ behavior: 'smooth' });
            } else {
                // 显示错误信息
                resultsContent.innerHTML = result.html;
            }
        } catch (error) {
            resultsContent.innerHTML = `
                <div class="alert alert-danger">
                    <strong>处理出错：</strong><br>
                    ${error.message}
                </div>
            `;
        }
    });
});
