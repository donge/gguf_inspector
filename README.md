# GGUF Inspector

gguf-inspector 是一个用 Rust 编写的、快速、无依赖的命令行工具，用于检查和展示 GGUF (Georgi Gerganov Universal Format) 模型文件的元数据。

它的核心设计理念是速度。通过只读取文件的元信息部分（文件头、元数据、张量信息），而完全跳过体积庞大的张量数据，它可以在几乎瞬间完成对数十 GB 大小的模型文件的分析。

---

## ✨ 主要功能

- **瞬时分析**：秒级打开并解析 GGUF 文件的元信息，无论文件有多大。
- **智能读取**：只读取文件的头部和元数据，完全不加载庞大的权重数据，I/O 开销极小。
- **零依赖**：编译后生成单个可执行文件，无需任何外部依赖或运行时，方便分发和使用。
- **功能丰富**：支持多种命令行参数，可以按需展示不同信息，如文件头、元数据或张量列表。
- **强大过滤**：可以根据关键字快速过滤你感兴趣的元数据或张量。
- **机器可读**：支持将所有信息以 JSON 格式输出，方便与其他脚本或程序集成。

---

## 🚀 安装与编译

编译 gguf-inspector 非常简单，你只需要安装 Rust 工具链。

### 1. 克隆或创建项目

如果你是从零开始，可以按照之前的步骤创建项目。如果已有项目，请确保 `Cargo.toml` 和 `src/main.rs` 文件内容正确。

### 2. 编译项目

在项目根目录下，运行以下命令来编译生成一个优化后的可执行文件：

```bash
cargo build --release
```

### 3. 找到可执行文件

编译成功后，你可以在 `target/release/` 目录下找到名为 `gguf-inspector` (Windows下为 `gguf-inspector.exe`) 的可执行文件。你可以将它复制到你的系统路径下（如 `/usr/local/bin`）以便在任何地方调用。

---

## 💡 使用方法

gguf-inspector 的使用方式非常直观。

### 基本用法

直接跟上 GGUF 文件路径，将以人类可读的格式打印所有元信息。

```bash
./gguf-inspector /path/to/your/model.gguf
```

### 显示特定部分

你可以只显示你感兴趣的部分。

- 只看文件头：

  ```bash
  ./gguf-inspector /path/to/model.gguf --header
  ```

- 只看元数据：

  ```bash
  ./gguf-inspector /path/to/model.gguf --metadata
  ```

- 只看张量信息：

  ```bash
  ./gguf-inspector /path/to/model.gguf --tensors
  ```

### 过滤信息

快速找到你想要的内容。

- 过滤元数据 (例如，查找所有与 token 相关的键):

  ```bash
  ./gguf-inspector /path/to/model.gguf --metadata --filter-meta "token"
  ```

- 过滤张量 (例如，查找所有与 attention 相关的权重):

  ```bash
  ./gguf-inspector /path/to/model.gguf --tensors --filter-tensor "attention"
  ```

### JSON 输出

将所有元信息以 JSON 格式输出，非常适合脚本处理。

```bash
./gguf-inspector /path/to/model.gguf --json
```

你也可以将其重定向到文件：

```bash
./gguf-inspector /path/to/model.gguf --json > model_metadata.json
```


### 输出样例

```bash
--- 文件头 (Header) ---
  GGUF 版本: v3
  张量数量: 310
  元数据条目数: 28

--- 元数据 (Metadata) ---
  - general.architecture: String("qwen3")
  - general.basename: String("Qwen3")
  - general.file_type: U32(7)
  - general.finetune: String("Instruct")
  - general.name: String("Qwen3 0.6B Instruct")
  - general.quantization_version: U32(2)
  - general.size_label: String("0.6B")
  - general.type: String("model")
  - qwen3.attention.head_count: U32(16)
  - qwen3.attention.head_count_kv: U32(8)
  - qwen3.attention.key_length: U32(128)
  - qwen3.attention.layer_norm_rms_epsilon: F32(1e-6)
  - qwen3.attention.value_length: U32(128)
  - qwen3.block_count: U32(28)
  - qwen3.context_length: U32(40960)
  - qwen3.embedding_length: U32(1024)
  - qwen3.feed_forward_length: U32(3072)
  - qwen3.rope.freq_base: F32(1000000.0)
  - tokenizer.chat_template: String("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for index in range(ns.last_query_index, -1, -1) %}\n    {%- set message = messages[index] %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not('<tool_response>' in message.content and '</tool_response>' in message.content) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}")
  - tokenizer.ggml.add_bos_token: Bool(false)
  - tokenizer.ggml.bos_token_id: U32(151643)
  - tokenizer.ggml.eos_token_id: U32(151645)
  - tokenizer.ggml.merges: [Array of 151387 items, first 8: [String("Ġ Ġ"), String("ĠĠ ĠĠ"), String("i n"), String("Ġ t"), String("ĠĠĠĠ ĠĠĠĠ"), String("e r"), String("ĠĠ Ġ"), String("o n")]...]
  - tokenizer.ggml.model: String("gpt2")
  - tokenizer.ggml.padding_token_id: U32(151643)
  - tokenizer.ggml.pre: String("qwen2")
  - tokenizer.ggml.token_type: [Array of 151936 items, first 8: [I32(1), I32(1), I32(1), I32(1), I32(1), I32(1), I32(1), I32(1)]...]
  - tokenizer.ggml.tokens: [Array of 151936 items, first 8: [String("!"), String("\""), String("#"), String("$"), String("%"), String("&"), String("'"), String("(")]...]

--- 张量信息 (Tensors) ---
名称                                                 | 类型       | 形状                   | 尺寸 (约)
----------------------------------------------------------------------------------------------------
output_norm.weight                                 | F32      | [1024]               | 4.00 KiB
token_embd.weight                                  | Q8_0     | [1024, 151936]       | 148.38 MiB
blk.0.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.0.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.0.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.0.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.0.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.0.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.0.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.0.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.0.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.0.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.0.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.1.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.1.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.1.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.1.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.1.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.1.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.1.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.1.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.1.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.1.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.1.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.2.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.2.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.2.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.2.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.2.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.2.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.2.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.2.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.2.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.2.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.2.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.3.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.3.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.3.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.3.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.3.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.3.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.3.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.3.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.3.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.3.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.3.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.4.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.4.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.4.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.4.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.4.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.4.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.4.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.4.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.4.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.4.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.4.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.5.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.5.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.5.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.5.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.5.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.5.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.5.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.5.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.5.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.5.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.5.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.6.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.6.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.6.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.6.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.6.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.6.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.6.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.6.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.6.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.6.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.6.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.7.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.7.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.7.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.7.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.7.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.7.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.7.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.7.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.7.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.7.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.7.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.8.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.8.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.8.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.8.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.8.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.8.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.8.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.8.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.8.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.8.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.8.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.9.attn_k.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.9.attn_k_norm.weight                           | F32      | [128]                | 512 B
blk.9.attn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.9.attn_output.weight                           | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.9.attn_q.weight                                | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.9.attn_q_norm.weight                           | F32      | [128]                | 512 B
blk.9.attn_v.weight                                | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.9.ffn_down.weight                              | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.9.ffn_gate.weight                              | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.9.ffn_norm.weight                              | F32      | [1024]               | 4.00 KiB
blk.9.ffn_up.weight                                | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.10.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.10.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.10.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.10.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.10.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.10.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.10.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.10.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.10.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.10.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.10.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.11.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.11.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.11.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.11.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.11.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.11.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.11.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.11.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.11.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.11.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.11.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.12.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.12.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.12.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.12.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.12.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.12.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.12.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.12.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.12.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.12.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.12.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.13.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.13.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.13.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.13.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.13.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.13.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.13.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.13.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.13.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.13.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.13.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.14.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.14.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.14.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.14.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.14.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.14.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.14.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.14.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.14.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.14.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.14.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.15.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.15.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.15.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.15.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.15.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.15.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.15.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.15.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.15.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.15.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.15.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.16.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.16.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.16.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.16.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.16.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.16.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.16.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.16.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.16.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.16.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.16.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.17.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.17.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.17.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.17.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.17.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.17.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.17.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.17.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.17.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.17.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.17.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.18.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.18.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.18.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.18.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.18.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.18.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.18.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.18.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.18.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.18.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.18.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.19.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.19.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.19.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.19.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.19.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.19.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.19.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.19.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.19.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.19.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.19.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.20.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.20.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.20.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.20.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.20.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.20.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.20.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.20.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.20.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.20.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.20.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.21.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.21.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.21.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.21.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.21.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.21.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.21.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.21.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.21.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.21.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.21.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.22.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.22.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.22.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.22.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.22.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.22.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.22.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.22.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.22.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.22.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.22.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.23.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.23.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.23.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.23.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.23.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.23.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.23.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.23.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.23.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.23.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.23.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.24.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.24.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.24.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.24.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.24.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.24.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.24.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.24.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.24.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.24.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.24.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.25.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.25.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.25.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.25.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.25.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.25.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.25.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.25.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.25.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.25.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.25.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.26.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.26.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.26.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.26.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.26.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.26.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.26.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.26.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.26.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.26.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.26.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.27.attn_k.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.27.attn_k_norm.weight                          | F32      | [128]                | 512 B
blk.27.attn_norm.weight                            | F32      | [1024]               | 4.00 KiB
blk.27.attn_output.weight                          | Q8_0     | [2048, 1024]         | 2.00 MiB
blk.27.attn_q.weight                               | Q8_0     | [1024, 2048]         | 2.00 MiB
blk.27.attn_q_norm.weight                          | F32      | [128]                | 512 B
blk.27.attn_v.weight                               | Q8_0     | [1024, 1024]         | 1.00 MiB
blk.27.ffn_down.weight                             | Q8_0     | [3072, 1024]         | 3.00 MiB
blk.27.ffn_gate.weight                             | Q8_0     | [1024, 3072]         | 3.00 MiB
blk.27.ffn_norm.weight                             | F32      | [1024]               | 4.00 KiB
blk.27.ffn_up.weight                               | Q8_0     | [1024, 3072]         | 3.00 MiB
```



---

## 📄 许可证

本项目采用 MIT 许可证。


