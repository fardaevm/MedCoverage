
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from typing import List

from src.rag_pipeline import RAGPipeline
from src.create_parser import create_parser
from src.rag import Datastore, Indexer, Retriever, ResponseGenerator



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "sample_data" / "source"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "sample_questions.json"
ALLOWED_EXTS = {".pdf"}


def get_files_in_directory(source_path: Path) -> List[str]:
    source_path = source_path.expanduser().resolve()
    if source_path.is_file():
        return [str(source_path)] if source_path.suffix.lower() in ALLOWED_EXTS else []

    return [
        str(p)
        for p in source_path.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in ALLOWED_EXTS
    ]


def create_pipeline() -> RAGPipeline:
    ds = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=ds)
    rg = ResponseGenerator()
    return RAGPipeline(ds, indexer, retriever, rg)


def main():
    parser = create_parser()
    args = parser.parse_args()

    pipeline = create_pipeline()

    source_path = Path(args.path).expanduser() if args.path else DEFAULT_SOURCE_PATH
    eval_path = Path(args.eval_file).expanduser() if args.eval_file else DEFAULT_EVAL_PATH

    if args.command in {"reset", "run"}:
        print("🗑️ Resetting datastore...")
        pipeline.reset()

    if args.command in {"add", "run"}:
        docs = get_files_in_directory(source_path)
        if not docs:
            raise SystemExit(f"No PDFs found in: {source_path}")
        print(f"📚 Indexing {len(docs)} PDFs from: {source_path}")
        pipeline.add_documents(docs)

    if args.command == "query":
        print(pipeline.process_query(args.prompt))

    if args.command in {"evaluate", "run"}:
        if not eval_path.exists():
            raise SystemExit(f"Eval file not found: {eval_path}")
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        pipeline.evaluate(data)  # only if you implemented evaluate


if __name__ == "__main__":
    main()
