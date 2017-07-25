# coding: utf-8

import regex


def tf(x: str, y: str) -> complex:
    return count(x, y) / pow(2, len(y))


def idf(docs: list, x: str) -> complex:
    from math import log10
    return log10(len(docs) / sum(bin_tf(x, y) for y in docs))


def bin_tf(x: str, y: str) -> int:
    return 1 if appartient(x, y) else 0


def tf_idf(x: str, y: str, docs: list) -> complex:
    return tf(x, y) * idf(docs, x)


def count(x: str, y: str, overlapped: bool=False) -> int:
    return len(regex.findall(x, y, overlapped=overlapped))


def appartient(x: str, y: str, overlapped: bool=False) -> bool:
    return True if regex.search(x, y, overlapped=overlapped) else False


def main():
    print()

if __name__ == '__main__':
    main()
