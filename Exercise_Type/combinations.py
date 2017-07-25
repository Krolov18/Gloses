# coding: utf-8


def combinations(n):
    import itertools
    yield ()
    for i in range(1, n):
        for x in itertools.combinations(range(n), i):
            yield x


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="combinations",
        description="générer les combinaisons à partir d'un nombre"
    )

    parser.add_argument(
        'n',
        type=int
    )

    args = parser.parse_args()

    for x in combinations(args.n):
        print(",".join(x), file=sys.stdout)


def main2():
    print(list(combinations(3)))


if __name__ == '__main__':
    main2()
