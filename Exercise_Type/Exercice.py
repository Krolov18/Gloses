# coding: utf-8


class Exercice(object):
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
    from sys import stderr
    if debug:
        print("distrib: ", struc.values(), file=stderr)
    for k, v in zip(y, x):
        struc[k]['buffer'] += v
    return struc


def retire_soluce(y: range, struc: dict):
    for k in y:
        struc[k]['buffer'] = struc[k]['buffer'][:-1]
    return struc


def initialise(taille: int):
    return {i: {"buffer": str(), "posss": None} for i in range(1, taille)}


def procedure_mokrwaze(taille: int, bdd: dict, debug=False):
    from sys import stderr
    struc = initialise(taille=taille+1)
    if debug:
        print("struc", struc, sep=': ', file=stderr)
    seq_p = range(2, taille+1, 2)
    seq_i = range(1, taille+1, 2)
    if debug:
        print(list(seq_p), list(seq_i), sep=' ----- ', file=stderr)
    i = 1
    while i <= taille:
        if not struc.get(i).get('posss'):
            struc[i]['posss'] = filter(
                lambda x: x.startswith(struc.get(i).get('buffer')),
                bdd
            )
        answer = next(struc.get(i).get('posss'), None)
        if debug:
            print(i, struc.get(i), file=stderr)
        if bdd.get(struc.get(i).get('buffer'), False):
            if est_pair(i):
                distribue_soluce(answer, seq_i, struc)
            else:
                distribue_soluce(answer, seq_p, struc)
            yield struc

        # if debug:
        #     print(i, answer, struc, sep=' || ', file=stderr)
        while answer is not None:
            if est_pair(i):
                struc = distribue_soluce(answer, seq_i, struc, False)
                if debug:
                    print(i, answer, struc, sep=' || ', file=stderr)
                i += 1
                break
            elif not est_pair(i):
                struc = distribue_soluce(answer, seq_p, struc, False)
                if debug:
                    print(i, answer, struc, sep=' || ', file=stderr)
                i += 1
                break
            answer = next(struc.get(i).get('posss'), None)
        else:
            i -= 1
            if est_pair(i):
                struc = retire_soluce(seq_i, struc)
            else:
                struc = retire_soluce(seq_p, struc)


def est_pair(x: int):
    return True if not (x % 2) else False


def main():
    import sqlite3
    import itertools
    # import yaml
    # import re
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
        taille = 3
        cursor = connect.cursor()
        cursor.execute('SELECT phon FROM Lexique WHERE length(phon)=?', (taille,))
        bdd = set([neutralise(x, symbs) for x in itertools.chain(*cursor.fetchall())])
        bdd = dict(zip(bdd, [True]*len(bdd)))
        tmp = procedure_mokrwaze(taille=taille, bdd=bdd, debug=True)
        next(tmp)

if __name__ == '__main__':
    main()
