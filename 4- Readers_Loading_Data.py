from pathlib import Path
from datetime import datetime

from llama_index.core import VectorStoreIndex, Document, Settings, SimpleDirectoryReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

load_dotenv()


def get_file_metadata(file_path: str) -> dict:
    """
    Custom metadata function for SimpleDirectoryReader.
    It receives a file path and returns a dictionary of metadata
    that will be attached to every Document created from that file.
    """
    path = Path(file_path)
    stat = path.stat()

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_extension": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "source": "local_filesystem",
    }


Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
Settings.llm = GoogleGenAI(model="gemini-3.5-flash-lite")

documents = SimpleDirectoryReader(
    input_dir="sample_data",      # Directory to read files from.
    input_files=None,             # Explicit list of file paths to read; if provided, input_dir is ignored.
    exclude=None,                 # List of file paths or patterns to exclude from reading.
    exclude_hidden=True,          # Whether to ignore hidden files/directories.
    errors="ignore",              # Error handling mode for decoding: "ignore", "strict", etc.
    recursive=False,              # Whether to read files from subdirectories recursively.
    encoding="utf-8",             # Text encoding used when reading files.
    filename_as_id=False,         # If True, use the file name as the document ID.
    required_exts=None,           # Only read files with these extensions, e.g. [".txt", ".pdf"].
    file_extractor=None,          # Dict mapping file extensions to custom file readers.
    num_files_limit=None,         # Maximum number of files to read; None means no limit.
    file_metadata=get_file_metadata,  # Callable that returns a metadata dict for each file path.
    raise_on_error=False,         # If True, raise an exception when a file cannot be read.
    fs=None,                      # fsspec filesystem to use; None means local filesystem.
).load_data(
    show_progress=False,          # Whether to show a progress bar while loading files.
    num_workers=None,             # Number of worker threads/processes; None lets the reader decide.
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("what frameworks is suitable for building RAG applications?")

print("ANSWER:")
print(response)

print("\n" + "=" * 80)
print("SOURCES:")
print("=" * 80)

for i, node in enumerate(response.source_nodes, 1):
    print(f"\n--- Source {i} ---")

    print("File:")
    print(node.metadata.get("file_name"))

    print("\nRetrieved text:")
    print(node.text)

    print("\nMetadata:")
    print(node.metadata)

    print("-" * 80)