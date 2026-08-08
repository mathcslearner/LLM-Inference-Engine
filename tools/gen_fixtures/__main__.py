"""CLI: python -m gen_fixtures <subcommand> [--out DIR]."""

import argparse

from . import model_configs, qwen2_names, tiny_llama, tokenizer_vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gen_fixtures",
        description="Regenerate committed golden fixtures (tests/fixtures/). "
        "Deterministic: re-running produces byte-identical output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    tiny_llama.register(subparsers)
    qwen2_names.register(subparsers)
    tokenizer_vectors.register(subparsers)
    model_configs.register(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
