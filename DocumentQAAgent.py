from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import requests
import json
import gradio as gr

class Tool:
    """工具类：封装单个功能，便于Agent统一管理和调用"""
    def __init__(self, name: str, func: Callable, description: str, function_description: Dict[str, Any]):
        """初始化工具
        Args:
            name: 工具名称，如'document_loader'
            func: 工具对应的函数
            description: 工具功能描述
            function_description: 符合OpenAI function calling格式的函数描述
        """
        self.name = name
        self.func = func
        self.description = description
        self.function_description = function_description
    
    def run(self, *args, **kwargs):
        """执行工具函数
        Returns:
            工具函数的执行结果
        """
        return self.func(*args, **kwargs)

class FunctionCallDocumentQAAgent:
    """使用function call机制的文档问答Agent"""
    def __init__(self, api_key: str, api_url: str, model_name: str):
        """初始化Agent
        Args:
            api_key: 大模型API密钥
            api_url: API端点URL
            model_name: 使用的大模型名称
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.conversation_history = []  # 存储历史对话记录
        self.tools = self._initialize_tools()  # 初始化所有可用工具
        self.max_retries = 3  # 最大重试次数
        self.current_file = None  # 当前处理的文件路径
        self.context = {}  # 执行上下文

    def _initialize_tools(self) -> Dict[str, Tool]:
        """注册所有可用工具
        Returns:
            工具字典，键为工具名称，值为Tool对象
        """
        return {
            # 文档加载工具：支持PDF/TXT文件
            'document_loader': Tool(
                name='document_loader',
                func=self._load_document,
                description='加载PDF或TXT文档',
                function_description={
                    "name": "document_loader",
                    "description": "加载PDF或TXT文档内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文档文件路径"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            ),
            # 文本搜索工具：基于TF-IDF的相似度搜索
            'text_search': Tool(
                name='text_search',
                func=self._search_doc,
                description='在文档中搜索相关内容',
                function_description={
                    "name": "text_search",
                    "description": "在文档中搜索与问题相关的内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户问题"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回最相关的片段数量",
                                "default": 3
                            }
                        },
                        "required": ["question"]
                    }
                }
            ),
            # 生成最终回答工具
            'generate_answer': Tool(
                name='generate_answer',
                func=self._generate_answer,
                description='根据相关文本生成最终回答',
                function_description={
                    "name": "generate_answer",
                    "description": "根据文档中的相关内容生成对问题的最终回答",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户问题"
                            }
                        },
                        "required": ["question"]
                    }
                }
            )
        }

    def _load_document(self, file_path: str) -> str:
        """加载文档内容的具体实现
        Args:
            file_path: 文档文件路径
        Returns:
            文档文本内容或错误信息
        """
        try:
            # PDF文件处理
            if file_path.endswith('.pdf'):
                reader = PdfReader(file_path)
                # 提取所有页面文本并用换行符连接
                text = '\n'.join([page.extract_text() for page in reader.pages])
                self.context['doc_text'] = text  # 保存到上下文
                return f"成功加载PDF文档，共{len(reader.pages)}页，{len(text)}字符。"
            # TXT文件处理
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.context['doc_text'] = text  # 保存到上下文
                    return f"成功加载TXT文档，共{len(text)}字符。"
            return "不支持的文件格式"
        except Exception as e:
            return f"文档加载失败: {str(e)}"

    def _search_doc(self, question: str, top_k: int = 3) -> str:
        """基于TF-IDF的文本搜索实现
        Args:
            question: 用户问题
            top_k: 返回最相关的top_k个片段
        Returns:
            搜索结果描述
        """
        if 'doc_text' not in self.context:
            return "请先加载文档"
            
        doc_text = self.context['doc_text']
        # 1. 文档预处理：按句号分割并过滤空句子
        sentences = [s.strip() for s in doc_text.split('.') if s.strip()]
        if not sentences:
            return "文档中无有效内容"
        
        try:
            # 2. 使用TF-IDF向量化文本
            vectorizer = TfidfVectorizer()
            # 将文档句子和问题一起向量化（最后一个是问题）
            tfidf_matrix = vectorizer.fit_transform(sentences + [question])
            
            # 3. 计算相似度
            doc_vectors = tfidf_matrix[:-1]  # 所有文档句子的向量
            question_vector = tfidf_matrix[-1]  # 问题的向量
            # 计算余弦相似度
            similarities = doc_vectors.dot(question_vector.T).toarray().flatten()
            
            # 4. 获取相似度最高的top_k个句子索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_texts = [sentences[i] for i in top_indices]
            
            # 保存到上下文
            self.context['relevant_texts'] = relevant_texts
            self.context['question'] = question
            
            return f"已找到{len(relevant_texts)}个相关文本片段。"
        except Exception as e:
            return f"搜索失败: {str(e)}"

    def _generate_answer(self, question: str) -> str:
        """生成最终回答
        Args:
            question: 用户问题
        Returns:
            生成的回答
        """
        if 'relevant_texts' not in self.context:
            return "请先搜索相关内容"
            
        relevant_texts = self.context['relevant_texts']
        # 构造提示词
        context_text = "\n".join(relevant_texts)
        prompt = f"""基于以下文档内容回答问题:
                   文档内容: {context_text}
                   问题: {question}
                   请提供准确、简洁的回答。"""
                   
        # 调用LLM生成回答
        return self._ask_llm_direct(prompt)

    def _ask_llm_direct(self, prompt: str) -> str:
        """直接调用LLM生成回答（不使用function call）
        Args:
            prompt: 提示词
        Returns:
            LLM生成的回答
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return f"API调用失败: {response.status_code} - {response.text}"
        except Exception as e:
            return f"API调用异常: {str(e)}"

    def _ask_llm_with_functions(self, messages):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 根据智谱API文档调整请求格式
            data = {
                "model": self.model_name,
                "messages": messages,
                "tools": [{"type": "function","function": tool.function_description} for tool in self.tools.values()],
                "tool_choice": "auto"
            }
            
            # 添加请求和响应的详细日志
            print("发送请求:", json.dumps(data, ensure_ascii=False, indent=2))
            response = requests.post(self.api_url, headers=headers, json=data, timeout=15)
            
            print("API响应状态码:", response.status_code)
            if response.status_code == 200:
                result = response.json()
                print("API响应内容:", json.dumps(result, ensure_ascii=False, indent=2))
                return result
            
            print(f"API调用失败: {response.status_code} - {response.text}")
            return {"error": f"API调用失败: {response.status_code}"}
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return {"error": f"API调用异常: {str(e)}"}

    def process_query(self, file_path: str, question: str) -> str:
        """处理用户查询的主流程
        Args:
            file_path: 文档文件路径
            question: 用户问题
        Returns:
            生成的回答文本
        """
        # 重置上下文
        self.context = {}
        self.current_file = file_path
        
        # 初始化对话历史
        messages = [{
            "role": "user", 
            "content": f"我有一个文档位于'{file_path}'，我想问关于这个文档的问题: {question}"
        }]
        
        # 最大对话轮次
        max_turns = 10
        
        for turn in range(max_turns):
            print(f"对话轮次 {turn+1}/{max_turns}")
            # 调用LLM
            response = self._ask_llm_with_functions(messages)
            
            # 检查是否有错误
            if "error" in response:
                return f"处理失败: {response['error']}"
                
            # 获取LLM回复
            message = response["choices"][0]["message"]
            print(f"LLM回复: {json.dumps(message, ensure_ascii=False, indent=2)}")
            
            # 添加到对话历史
            messages.append(message)
            
            # 检查是否有函数调用
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    try:
                        function_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        
                        if function_name in self.tools:
                            tool = self.tools[function_name]
                            result = tool.run(**arguments)
                            messages.append({
                                "role": "tool",  # 智谱使用tool而非function
                                "name": function_name,
                                "content": str(result)
                            })
                        else:
                            return f"未知函数: {function_name}"
                    except Exception as e:
                        return f"函数执行错误: {str(e)}"
            else:
                # 没有函数调用，返回LLM的回复
                return message.get("content", "无回复内容")
        
        return "处理超过最大轮次限制，请尝试简化问题"

# 示例使用
if __name__ == "__main__":
    # 初始化Agent（需替换为真实API密钥和支持function call的模型）
    agent = FunctionCallDocumentQAAgent(
        api_key="0fef2c07feb447b992e5f65cca6a5676.Ff3PxdqyPc08Ze4A",
        api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",  # 使用支持function call的API端点
        model_name="glm-4-flash"  # 使用完整版而非flash版本
    )
    
    # 创建Gradio界面处理函数
    def handle_query(file, question):
        if file is None:
            return "请先上传文档文件"
        return agent.process_query(file.name, question)
    
    # 创建并启动Gradio Web界面
    iface = gr.Interface(
        fn=handle_query,
        inputs=[gr.File(label="上传文档"), gr.Textbox(label="问题")],
        outputs="text",
        title="文档问答助手（Function Call版）"
    )
    iface.launch(server_port=8084)

''' 当用户提交问题时，执行流程如下：
    1. 调用process_query(file_path, question)
    2. 初始化对话，向LLM发送包含文件路径和问题的消息
    3. LLM分析问题，决定调用哪个函数(function_call)
    4. 通常会先调用document_loader加载文档
    5. 然后调用text_search在文档中搜索相关内容：
       - 将文档按句号分割
       - 使用TF-IDF向量化文本
       - 计算问题与各文本片段的相似度
       - 选取相似度最高的片段
    6. 最后调用generate_answer生成回答：
       - 将相关文本片段组合成上下文
       - 构造提示词
       - 调用LLM生成最终回答
    7. 返回生成的回答给用户
    遵循"用户输入→LLM决策→工具执行→结果回传→LLM响应"的流程
'''
'''
LLM决定调用哪个函数的过程是基于：
函数描述：了解每个函数的功能和参数
当前上下文：分析已有信息和任务需求（LLM通过对话历史跟踪已完成的步骤）
3. 内在能力：利用自身的推理能力匹配任务需求与可用函数
这种机制使得系统能够像人类一样思考："为了完成这个任务，我现在需要做什么？我有哪些工具可用？"，
从而实现真正的动态决策。
'''