from typing import List
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dotenv import load_dotenv
load_dotenv()


class CustomDatabaseReader(BaseReader):
    """
    A custom Reader for reading data from SQL databases.
    This class uses SQLAlchemy to connect and execute queries.
    """
    def __init__(
        self,
        connection_string: str,
    ):
        """
        Args:
            connection_string: Database connection string.
                Example for PostgreSQL: "postgresql://user:pass@localhost:5432/mydb"
        """
        self.connection_string = connection_string

        # Create an Engine to manage database connections.
        self.engine: Engine = create_engine(connection_string)

    def load_data(self, query: str) -> List[Document]:
        """
        Execute the query and convert each row into a Document.
        Args:
            query: SQL query to run.
        Returns:
            A list of Document objects.
        """
        documents = []

        # Use a context manager to automatically close the connection.
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()  # Column names.

            for row in result:
                # Convert the row into a dictionary.
                row_dict = dict(zip(columns, row))

                # Build the main Document text from all columns.
                # You can customize this section based on your needs.
                text_content = "\n".join(
                        f"{col}: {val}" for col, val in row_dict.items()
                    )

                # Build metadata.
                metadata = {
                    "source": self.connection_string,  # Origin of the data.
                    "query": query,                    # The query that produced this row.
                }

                documents.append(
                    Document(
                        text=text_content,     # The text content of the Document.
                        metadata=metadata,     # Metadata attached to the Document.
                    )
                )

        return documents

Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
Settings.llm = GoogleGenAI(model="gemini-3.5-flash-lite")

connection_string = "postgresql://postgres:admin@localhost:5432/AdventureWorks"
query = 'SELECT fullname, "Occupation" FROM "CustomerLookup" LIMIT 10'

documents = CustomDatabaseReader(connection_string=connection_string).load_data(query=query)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the occupation of Morgan?")

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