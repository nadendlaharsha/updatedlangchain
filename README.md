# 🦜🔗 LangChain Essentials: Hands-on Modules

Welcome to **`updatedlangchain`** — a structured, hands-on repository demonstrating core **LangChain** concepts for building intelligent, tool-aware, and controllable AI agents.

---

## 📂 Repository Structure

The project is organized into 5 sequential Jupyter Notebook modules:

```text
updatedlangchain/
├── 1-lanchainintro.ipynb
├── 2-models&tools.ipynb
├── 3-messages.ipynb
├── 4-structured_output.ipynb
└── 5-Middleware.ipynb
```

---

## 📚 Module Overview & Key Concepts

### 1️⃣ `1-lanchainintro.ipynb` — Introduction & Agent Setup
* **Core Concepts**: LangChain Agent creation & API key configuration.
* **Key Topics**:
  * Setting up provider environment variables (`GROQ_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
  * Creating agents using `create_agent`.
  * Running basic conversational agent workflows.

---

### 2️⃣ `2-models&tools.ipynb` — Model Initialization & Custom Tools
* **Core Concepts**: Multi-provider LLM initialization and function calling.
* **Key Topics**:
  * Unified model setup with `init_chat_model()` across providers (Groq, Google Gemini, OpenAI).
  * Defining custom tools using the `@tool` decorator.
  * Binding tools to LLMs for automated tool call generation and execution.

---

### 3️⃣ `3-messages.ipynb` — Messages & Conversation State
* **Core Concepts**: LangChain standard message format and role management.
* **Key Topics**:
  * **SystemMessage**: Setting system context and behavioral guardrails.
  * **HumanMessage**: Capturing user inputs.
  * **AIMessage**: Processing model responses, reasoning content, and tool call requests.
  * **ToolMessage**: Returning tool execution outputs back to the model.
  * Tracking token consumption and response metadata.

---

### 4️⃣ `4-structured_output.ipynb` — Structured Outputs with Pydantic
* **Core Concepts**: Enforcing strict schema outputs from LLMs.
* **Key Topics**:
  * Defining data schemas using **Pydantic** models.
  * Enforcing typed JSON responses using `.with_structured_output()`.
  * Extracting structured entities for automated data processing pipelines.

---

### 5️⃣ `5-Middleware.ipynb` — Agent Middleware & Governance
* **Core Concepts**: Intercepting and controlling agent behavior during execution.
* **Key Topics**:
  * **Summarization Middleware**: Automatically compressing long conversation histories when approaching token or message limits while preserving context.
  * **Human-in-the-Loop (HITL) Middleware**: Intercepting high-stakes tool executions (e.g., sending emails) for human review with **Approve**, **Edit**, or **Reject** control flows.

---

## 🛠️ Requirements & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd langchainupdated
   ```

2. **Set up virtual environment & install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## 🚀 Usage

Open any notebook in VS Code or Jupyter Lab to explore the step-by-step examples:

```bash
jupyter lab
```
