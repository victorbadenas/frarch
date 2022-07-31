from glob import glob
from typing import List


def list_experiments() -> List[str]:
    return list(glob("./examples/*"))


if __name__ == "__main__":
    print(list_experiments())
