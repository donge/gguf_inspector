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

---

## 📄 许可证

本项目采用 MIT 许可证。


