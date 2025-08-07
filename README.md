项目名称： 基于LangChain的本地文档问答助手

项目简介：设计并实现基于Agent架构的文档问答系统，通过Function Calling机制协调文档加载语义检索答案，实现PDF/TXT文件的加载检索问答全流程。

核心创新：工具化架构：封装文档加载（PyPDF2）、TF-IDF检索（sklearn）、GLM-4问答生成三大可扩展模块；动态决策：利用GLM-4的Function Calling自动选择工具链，减少人工规则编码。上下文管理：通过self.context字典实现跨工具数据共享，支持多轮次对话状态保持。

技术实现：开发Agent核心类（FunctionCallDocumentQAAgent），实现工具动态调度与异常重试机制；构建TF-IDF增强检索模块；适配智谱API协议(非标准OpenAI格式)，完成请求封装与tool_calls响应解析；搭建Gradio交互界面，支持文档上传与实时问答。
