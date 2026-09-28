"""Unpruned model: the reference for every pruned arm."""
import config
from common import core


def main():
    core.setup()
    tok = core.load_tokenizer()
    core.run_utility_eval("baseline", config.MODEL_ID)
    core.generate_safety("baseline", config.MODEL_ID, tok, core.load_safety_benchmarks())


if __name__ == "__main__":
    main()
