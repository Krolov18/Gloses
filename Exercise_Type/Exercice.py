# coding: utf-8
import typing
from operator import add

class Exercise(object):
    def __init__(self):
        pass
        # champs = ('SA', 'SAC', 'MC', 'MCV', 'MCH', 'RX', "RXC", 'TXT')

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def verifier(self, dico): pass


def phonoku(bdd, taille=10, output=""):
    import itertools
    import sys
    for perm in itertools.permutations(iterable=bdd, r=taille):
        i = 0
        cml = "".join((y[i] for y in perm))
        while cml in bdd:
            print(cml)
            i += 1
            cml = "".join((y[i] for y in perm))
            if i + 1 == taille:
                print(perm, file=open(output, 'w'), sep='\n')
                print(perm, file=sys.stderr)
                break


def neutralise(chaine, symbs):
    for x in set(chaine):
        chaine = chaine.replace(x, symbs.get(x, x))
    return chaine


def distribue_soluce(x: str, y: range, struc: dict, debug=False):
    for k, v in zip(y, x):
        struc[k]['buffer'] += v
    return struc


def retire_soluce(y: range, struc: dict):
    for k in y:
        struc[k]['buffer'] = struc[k]['buffer'][:-1]
    return struc


def initialise(taille: int):
    return {i: {"buffer": str(), "posss": None} for i in range(1, taille)}


def procedure_mokrwaze(taille: typing.Union[int, tuple], bdd: list, debug=False):
    from sys import stderr

    if isinstance(taille, int):
        taille = (taille, taille)
    elif isinstance(taille, tuple) and (len(taille) != 2):
        raise Exception(
            'Attention, on ne travaille qu\'en bidimentionnel, donc, soit un entier, soit un couple d\'entier'
        )
    length = add(*taille)
    length_1 = length + 1
    struc = initialise(length_1)
    print(struc)
    seq_p = range(2, length_1, 2)
    seq_i = range(1, length_1, 2)
    i = 1
    while i <= length:
        if not struc.get(i).get('posss'):
            struc[i]['posss'] = filter(lambda x: x.startswith(struc.get(i).get('buffer')), bdd)
        answer = next(struc.get(i).get('posss'), None)
        print(answer)
        while answer is not None:
            distribue_soluce(answer, seq_i if est_pair(i) else seq_p, struc, True)
            if i == length:
                yield struc
                retire_soluce(seq_i if est_pair(i) else seq_p, struc)
                i -= 1
                answer = next(struc.get(i).get('posss'), None)
            else:
                i += 1
        i -= 1
        retire_soluce(seq_i if est_pair(i) else seq_p, struc)


def procedure_mokrwaze_2(taille, bdd, debug):
    from sys import stderr
    from copy import deepcopy
    from codecs import open

    sortie = open('phonoku_{}.txt'.format(taille), 'a', 'utf-8')

    length_bdd = len(bdd)
    length = taille + taille
    length_1 = length + 1
    struc = initialise(length_1)
    seq_i = range(1, length_1, 2)
    seq_p = range(2, length_1, 2)
    i = 1
    struc[i]['posss'] = filter(lambda x: x.startswith(struc.get(i).get('buffer')), bdd)
    prec = None
    cur = None
    print('RESTE\tCOURANT\tLENGTH', file=stderr)
    while i <= length:
        if (i == 1) and (struc.get(i).get('posss') is None):
            break
        answer = next(struc.get(i).get('posss'), None)
        if i == 1:
            length_bdd -= 1
            print('{}\t{}\t{}'.format(length_bdd, answer, taille), file=stderr)
        while answer is not None:
            distribue_soluce(answer, seq_i if est_pair(i) else seq_p, struc)
            if i == length:
                print(*[struc.get(i).get('buffer') for i in seq_p], sep=";", end="\n", file=sortie)
                # print(*[struc.get(i).get('buffer') for i in seq_p], sep=";", end="\n")
                retire_soluce(seq_i if est_pair(i) else seq_p, struc)
            else:
                i += 1
            if struc.get(i).get('posss') is None:
                struc[i]['posss'] = filter(lambda x: x.startswith(struc.get(i).get('buffer')), bdd)
            answer = next(struc.get(i).get('posss'), None)
        struc[i]['posss'] = None
        if i > 1:
            i -= 1
        retire_soluce(seq_i if est_pair(i) else seq_p, struc)


def est_pair(x: int):
    return True if not (x % 2) else False


def main():
    import sqlite3
    import itertools
    # import yaml
    import re
    # import sys

    symbs = {
        '2': '2',
        '5': 'ê',
        '8': 'y',
        '9': '2',
        '@': 'â',
        'E': 'e',
        'G': 'G',
        'N': 'N',
        'O': 'o',
        'R': 'R',
        'S': 'S',
        'Z': 'Z',
        'a': 'a',
        'b': 'b',
        'd': 'd',
        'e': 'e',
        'f': 'f',
        'g': 'g',
        'i': 'i',
        'j': 'j',
        'k': 'k',
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'o': 'o',
        'p': 'p',
        's': 's',
        't': 't',
        'u': 'u',
        'v': 'v',
        'w': 'w',
        'y': 'y',
        'z': 'z',
        '§': 'ô',
        '°': '6'
    }

    with sqlite3.connect('../../Lexiques/Lexique381/Lexique.db') as connect:
        taille = 4
        cursor = connect.cursor()
        cursor.execute('SELECT phon FROM Lexique WHERE length(phon)=?', (taille,))
        bdd = sorted(set([neutralise(x, symbs) for x in itertools.chain(*cursor.fetchall())]))
        # print(list(filter(lambda y: y is not None, map(lambda x: x if re.search('.f.', x) else None, bdd))))
        # print(len(bdd))
        procedure_mokrwaze_2(taille=taille, bdd=bdd, debug=False)

if __name__ == '__main__':
    main()
