import os
import shutil
from typing import Tuple, List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class PrepareVectorDB:
    """
    Build a persistent Chroma vector DB from a plaintext transcript.

    Key features:
    - Passes GOOGLE_API_KEY explicitly to embeddings.
    - Version-proof Chroma usage: construct, add_documents, persist().
    - Rebuild option to delete old index cleanly.
    - Clear logs: document count, chunk count, and internal collection count.
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        vectordb_dir: str,
        collection_name: str,
        google_api_key: str,
        rebuild: bool = False,
        encoding: str = "utf-8",
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name
        self.google_api_key = google_api_key
        self.rebuild = rebuild
        self.encoding = encoding

    # ---------- validation & setup ----------

    def _validate_inputs(self) -> None:
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_API_KEY is missing. Add it to your .env file.")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ File not found: {self.file_path}")
        if not self.file_path.lower().endswith((".txt", ".text")):
            raise ValueError(f"❌ Unsupported file type for: {self.file_path} (expected .txt or .text)")

    def _prepare_dir(self) -> None:
        if self.rebuild and os.path.exists(self.vectordb_dir):
            print(f"🧹 Rebuild=True → deleting existing directory '{self.vectordb_dir}' ...")
            shutil.rmtree(self.vectordb_dir, ignore_errors=True)
        if not os.path.exists(self.vectordb_dir):
            os.makedirs(self.vectordb_dir, exist_ok=True)
            print(f"📁 Created directory '{self.vectordb_dir}'")

    # ---------- data loading & splitting ----------

    def _load_and_split(self) -> Tuple[List, List]:
        print(f"📖 Loading transcript: {self.file_path}")
        loader = TextLoader(self.file_path, encoding=self.encoding)
        docs_list = loader.load()
        print(f"✅ Loaded {len(docs_list)} document(s)")

        print(f"✂️ Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap}) ...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        doc_splits = splitter.split_documents(docs_list)
        print(f"✅ Created {len(doc_splits)} chunk(s)")
        return docs_list, doc_splits

    # ---------- build index ----------

    def run(self) -> None:
        self._validate_inputs()
        self._prepare_dir()

        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            api_key=self.google_api_key,
        )

        docs_list, doc_splits = self._load_and_split()

        print("🧠 Building Chroma vector store ...")
        vectordb = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.vectordb_dir,
            embedding_function=embeddings,
        )
        # add chunks & persist
        vectordb.add_documents(doc_splits)
        vectordb.persist()

        # Diagnostics
        try:
            internal_count = vectordb._collection.count()  # internal, but handy
        except Exception:
            internal_count = "N/A"

        print("✅ VectorDB built and persisted.")
        print(f"📦 Documents indexed: {len(docs_list)} | Chunks indexed: {len(doc_splits)}")
        print(f"🔢 Chroma internal count: {internal_count}")


if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    builder = PrepareVectorDB(
        file_path="cleaned_transcript.txt",
        chunk_size=2000,                 # safe default; adjust if needed
        chunk_overlap=200,               # modest overlap avoids redundant vectors
        embedding_model="models/text-embedding-004",
        vectordb_dir="vectordb",
        collection_name="chroma",
        google_api_key=GOOGLE_API_KEY,
        rebuild=True,                    # True = clean rebuild this run
        encoding="utf-8",
    )
    builder.run()
