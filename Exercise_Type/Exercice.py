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


def distribue_soluce(x: str, y: range, struc: dict):
    for k, v in zip(y, x):
        struc[k]['buffer'] += v
    return struc


def retire_soluce(y: range, struc: dict):
    for k in y:
        struc[k]['buffer'] = struc[k]['buffer'][:-1]
    return struc


def initialise(taille: int):
    return {i: {"buffer": str(), "posss": None} for i in range(1, taille+1)}


def procedure_mokrwaze(taille: int, bdd: dict, debug=False):
    struc = initialise(taille=taille)
    if debug:
        print("struc", struc, sep=': ')
    seq_p = range(2, taille+1, 2)
    seq_i = range(1, taille+1, 2)
    if debug:
        print(seq_p, seq_i, sep='-----')
    i = 1
    print('poire')
    while (i <= (taille * 2)) and bdd.get(struc.get(i).get('buffer'), False):
        print('camembert')
        if not struc.get(i).get('posss'):
            struc[i]['posss'] = filter(
                lambda x: x.startswith(struc.get(i).get('buffer')),
                bdd
            )
        answer = next(struc.get(i).get('posss'), None)
        while answer is not None:
            if est_pair(i):
                distribue_soluce(answer, seq_i, bdd)
                i += 1
                break
            elif not est_pair(i):
                distribue_soluce(answer, seq_p, bdd)
                i += 1
                break
            answer = next(struc.get(i).get('posss'), None)
        else:
            i -= 1
            if est_pair(i):
                retire_soluce(seq_i, bdd)
            else:
                retire_soluce(seq_p, bdd)
    print('banane')


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
        # cursor.execute('SELECT phon FROM Lexique WHERE length(phon)=?', (taille,))
        bdd = list(set([neutralise(x, symbs) for x in itertools.chain(*cursor.fetchall())]))
        print([x for x in bdd if x.startswith('')])
        print(list(filter(lambda x: x.startswith(''), bdd)))
        # for x in bdd:
        #    regex = [y+'.'*(len(x)-1) for y in x]
        #    i = 0
        #    for fs in itertools.product(*[[h for h in bdd if re.match(g, h)] for g in regex]):
        #        i += 1
        #        cml = "".join((y[i] for y in fs))
        #        if i + 1 == taille:
        #            print(fs, file=sys.stderr)
        #            break
        # print(bdd)
        # tmp = phonoku(bdd=bdd, output="phonoku_6")
        # list(tmp)


if __name__ == '__main__':
    main2()
