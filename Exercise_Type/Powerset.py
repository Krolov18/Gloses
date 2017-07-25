# coding: utf-8

import regex
import sqlite3

__description__ = """
    Programme qui permet de générer une partie du powerset d'une chaine
"""


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


class Variable(object):
    def __init__(self, var):
        self.a = var

    def __repr__(self):
        return str(self.a)

    def __str__(self):
        return repr(self)

    def __hash__(self): return 1


class OptimString(object):
    def __init__(self, seq):
        self.regex = seq
        self.liste = self.to_list(seq)

    @staticmethod
    def to_list(seq):
        # ..aa.. -> ['.', '.', 'a', 'a', '.', '.']
        pass


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="powerset",
        description=__description__
    )

    parser.add_argument(
        "chaine",
        type=list
    )

    parser.add_argument(
        "-k",
        "--positions",
        dest="positions"
    )

    args = parser.parse_args()

    print(powerset(args.chaine, args.positions.split(','), Variable('.'), Variable('.+'), True), file=sys.stdout)


def powerset_tostring(seq, ks, var_u, var_e, extend):
    return "".join(map(str, powerset(seq=seq, ks=ks, var_u=var_u, var_e=var_e, extend=extend)))


def main2():
    global seq, var_u, var_e, extend
    import itertools
    from Exercise_Type.combinations import combinations
    import multiprocessing
    seq = 'ant'
    var_u = Variable('.')
    var_e = Variable('.+')
    extend = True
    print(list(combinations(len(seq))))
    with multiprocessing.Pool(10) as proc:
        args = map(lambda x: (seq, x, var_u, var_e, extend), combinations(len(seq)))
        print(proc.starmap(powerset_tostring, args))


def dump(x: list, y: sqlite3.Cursor) -> None:
    create_tables = """
CREATE TABLE IF NOT EXISTS Weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER,
    feature_id INTEGER,
    value REAL,
    FOREIGN KEY (class_id) REFERENCES classes(id),
    FOREIGN KEY (feature_id) REFERENCES features(id)
)
CREATE TABLE IF NOT EXISTS Classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class STRING UNIQUE
)
CREATE TABLE IF NOT EXISTS Features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature STRING UNIQUE
)
"""
    cmd_id_val = """
SELECT value,id FROM Weights
WHERE class_id=EXISTS(SELECT id FROM Classes WHERE class=?)
AND
feature_id=EXISTS(SELECT id FROM Features WHERE feature=?)
"""
    cmd_update = "UPDATE Weights SET value=? WHERE id=?"
    y.executemany(cmd_id_val, map(lambda r: r[:-1], x))
    tmp = y.fetchall()
    y.executemany(cmd_update, tmp)


def powerset(seq: str, ks: list, var_u: Variable=Variable('.'), var_e: Variable=Variable('.+'), extend=False):
    d = [var_u if i in ks else x for i, x in enumerate(seq)]

    if extend:
        # ..a..
        z = list()
        v = None
        p = list()
        for x in d:
            if isinstance(x, Variable):
                # cas pour vider la zone v
                if p:
                    z.append("".join(p))
                    p.clear()
                # cas pour remplir la zone v
                if v is None:
                    v = x
                # cas pour .+ ou .*
                else:
                    v = var_e
            elif isinstance(x, str):
                if v is not None:
                    z.append(v)
                    v = None
                if not p:
                    p.append(x)
                else:
                    p.append(x)
        z.append(v) if v else z.append("".join(p))
        return z
    return d


def genrer_regexes(lexique381, bdlex):
    lexique_table = """CREATE TABLE IF NOT EXISTS Lexique (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {columns}

)"""
    regexes_table = """CREATE TABLE IF NOT EXISTS Regexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id INTEGER,
    regex STRING
)"""
    with open(name=lexique381, mode='r', encoding='utf-8') as lec:
        lexique_table = lexique_table.format(
            columns=",\n\t".join(
                [" ".join((x, 'STRING')) for x in map(lambda x: x.split('_')[-1], lec.readline().strip().split())]
            )
        )

    seq = "1_ortho	2_phon	3_lemme	4_cgram	5_genre	6_nombre	7_freqlemfilms2	8_freqlemlivres	9_freqfilms2	10_freqlivres	11_infover	12_nbhomogr	13_nbhomoph	14_islem	15_nblettres	16_nbphons	17_cvcv	18_p_cvcv	19_voisorth	20_voisphon	21_puorth	22_puphon	23_syll	24_nbsyll	25_cv-cv	26_orthrenv	27_phonrenv	28_orthosyll	29_cgramortho	30_deflem	31_defobs	32_old20	33_pld20	34_morphoder	35_nbmorph"


if __name__ == '__main__':
    main2()
