# README - RAG with LLaMA and ChromaDB

## 📖 Overview

This module (`chroma_db`) is part of the larger **[RAG-Finetuning-LLAMA-Model](https://github.com/hasnain393/RAG-Finetuning-LLAMA-Model)** project, showcasing how to build an end-to-end Retrieval-Augmented Generation pipeline using:

- 🧠 **HuggingFace LLaMA models** for answering queries  
- 🔍 **ChromaDB** for fast semantic search  
- 📄 **LangChain** for chaining retrieval + LLM responses  
- 🕸️ **AsyncWebCrawler** for scraping websites (e.g., haznain.com)

---

## 📁 Folder Structure

```
RAG-Finetuning-LLAMA-Model/
├── chroma_db/                  # Persisted Chroma vector store
├── output.txt                  # Extracted markdown content
├── main.py                     # RAG pipeline entry point
├── crawler.py                  # Async crawler script
├── requirements.txt            # All pip dependencies
├── README.md                   # Project documentation
└── utils/                      # (Optional) helper modules
```

---

## ⚙️ What This Module Does

1. Scrapes haznain.com using `crawl4ai.AsyncWebCrawler`  
2. Embeds content using `sentence-transformers/all-MiniLM-L6-v2`  
3. Stores vectors in a persistent ChromaDB store  
4. Loads a fine-tuned LLaMA model via HuggingFace  
5. Runs RetrievalQA using LangChain to answer user queries  

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You will also need:

- ✅ A valid HuggingFace token (set `HF_TOKEN` as an environment variable)  
- ✅ PyTorch with GPU support (`cuda` preferred)  
- ✅ Access to `meta-llama` models via HuggingFace  

---

## 🚀 How to Run

```bash
python main.py
```

It will:

- Crawl `haznain.com` and save content to `output.txt`  
- Load the LLaMA model for inference  
- Use ChromaDB to retrieve relevant context  
- Print out answers to your queries  

---

## 🔍 Example Query

```python
query = "What is the main focus of haznain.com?"
```

**Sample Output:**

```
Answer: The main focus of haznain.com is to share knowledge on software development, cloud architecture, DevOps, and integration through hands-on blog posts and tutorials.

Sources:
- output.txt (page N/A)
```

---

## 🔐 Access Notes

Ensure you have access to LLaMA models via HuggingFace:

1. Request access to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)  
2. Set your access token via:

```bash
huggingface-cli login
```

Or as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

---

## 🙌 Credits

Built by **[Hasnain Ahmed Shaikh](https://haznain.com)**  
A part of the hands-on book: _“From Code to Cloud: 10 Beginner Projects to Master Tech”_

---

## 📄 License

This project is licensed under the **MIT License** – feel free to use, fork, and contribute.

