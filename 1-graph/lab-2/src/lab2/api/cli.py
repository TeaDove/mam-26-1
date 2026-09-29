import argparse
import logging
from collections.abc import Callable

from lab2.api import stages
from lab2.util.settings import Settings

COMMANDS: dict[str, Callable[[Settings], None]] = {
    "chunk": stages.run_chunking,
    "clean": stages.run_cleaning,
    "normalize": stages.run_normalization,
    "tokenize": stages.run_tokenization,
    "vectorize": stages.run_vectorization,
    "export": stages.run_export,
    "compare": stages.run_compare,
    "figures": stages.run_figures,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab2")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    COMMANDS[args.command](Settings())


if __name__ == "__main__":
    main()
