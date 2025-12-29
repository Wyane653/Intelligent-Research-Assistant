## 🔍 核心实现

本项目采用模块化设计，核心流程清晰，职责分离明确，主要分为以下几个关键部分：

1.  **文档加载与预处理 (`src/document_processor.py`)**
    *   **职责**：负责原始文档的读取、解析和文本分割（分块），为后续的向量化做准备。
    *   **核心类**：`DocumentProcessor`
    *   **关键方法**：
        - `load_document()`: 加载并解析单个文档（支持 PDF, TXT, Markdown 等格式）。
        - `process_directory()`: 批量处理指定目录下的所有文档。
        - `chunk_documents()`: 将解析后的长文本按语义或固定大小进行分割，形成适合处理的文本块。

2.  **向量化与存储 (`src/vector_store.py`)**
    *   **职责**：将预处理后的文本块转换为向量嵌入，并存入向量数据库，实现高效的语义检索。
    *   **核心类**：`VectorStoreManager`
    *   **关键方法**：
        - `create_vector_store()`: 调用嵌入模型（如 OpenAI Embeddings）生成向量，并构建或更新向量数据库（如 Chroma）。

3.  **流程调度与主逻辑 (`main.py`)**
    *   **职责**：作为应用程序的入口，串联整个文档处理与加载流程。
    *   **核心类**：`IntelligentResearchAssistant`
    *   **关键方法**：
        - `load_documents()`: 调度 `DocumentProcessor` 和 `VectorStoreManager`，完成从原始文档到向量数据库的完整流水线。

4.  **智能体对话记忆 (`src/agent.py`)**
    *   **职责**：为基于LLM的智能体提供对话历史管理能力，使其能进行连贯的多轮对话。
    *   **核心类**：`ConversationBufferMemory`
    *   **关键机制**：该类在对话过程中自动保存和加载上下文，确保智能体能够理解并回应历史对话内容，从而维持强大的多轮对话能力。
  
   ```mermaid
graph TD
    A[原始文档] --> B[DocumentProcessor];
    B --> |load_document| C[解析文本];
    C --> |chunk_documents| D[文本块];
    D --> E[VectorStoreManager];
    E --> |create_vector_store| F[向量数据库];

    G[用户提问] --> H[智能体];
    F --> |语义检索| H;
    I[ConversationBufferMemory] --> |提供对话历史| H;
    H --> J[生成回答];
```
