# Sample data for LlamaIndex folder loading

This folder contains small, safe sample files for practicing `SimpleDirectoryReader` and a RAG index.

## Included formats

- `notes.txt` - plain text
- `knowledge_guide.md` - Markdown
- `faq.json` - JSON
- `products.csv` - CSV
- `web_page.html` - HTML
- `example.py` - Python source
- `nested/faq_about_rag.txt` - nested plain text

The files intentionally describe a fictional company, so they can safely be used in examples.

## Example

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("sample_data").load_data()
print(f"Loaded {len(documents)} documents")
for document in documents:
    print(document.metadata.get("file_path"))
```
