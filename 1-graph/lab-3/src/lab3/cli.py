import argparse
import logging

from lab3 import pipeline
from lab3.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab3")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "train", "queries", "retrieval", "rag"):
        commands.add_parser(name)
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--graph", default="clean")
    ask.add_argument("--retriever", default="gat")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    stages = {
        "features": pipeline.run_features,
        "train": pipeline.run_train,
        "queries": pipeline.run_queries,
        "retrieval": pipeline.run_retrieval,
        "rag": pipeline.run_rag,
    }
    if args.command == "ask":
        print(pipeline.run_ask(settings, args.question, args.graph, args.retriever))
        return
    stages[args.command](settings)


if __name__ == "__main__":
    main()
