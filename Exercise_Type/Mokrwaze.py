# coding: utf-8
from operator import add
from itertools import chain
from copy import deepcopy
import re
import numpy


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


def transpose(liste):
    if all(len(liste[0]) == len(x) for x in liste):
        return map(lambda x: "".join(x), map(list, numpy.transpose(list(map(list, liste)))))


def unifie_chaines(x, y):
    """
        Cette fonction fait la conjonction de deux expressions rationnelles.
            soit "a.." + "..r" donne "a.r"
        On se cantonne à ce que les deux chaines x et y soient de même longueur.
    :param x:
    :param y:
    :return:
    """
    c = str()
    if len(x) == len(y):
        for z, t in zip(x, y):
            if z == '.':
                c += t
            elif t == '.':
                c += z
            elif z == t:
                c += z
            else:
                return None
        return c


def sature_buffer(buffer, i):
    if len(buffer) != i:
        return buffer + (i-len(buffer))*'.'
    elif len(buffer) == i:
        return buffer
    else:
        raise IndexError()


def initialise(i, grid=None):
    sortie = dict()
    champs = {
        "regex": str(),
        "buffer": str(),
        "posss": None,
    }
    if grid:
        x = len(grid)
        y = len(grid[0])
        if add(x, y) == i:
            for (k, v) in zip(range(1, i + 1), chain(*zip(grid, transpose(grid)))):
                sortie[k] = deepcopy(champs)
                sortie[k]['regex'] = v
            return sortie
        elif x == i:
            for (k, v) in zip(range(1, i + 1), grid):
                sortie[k] = deepcopy(champs)
                sortie[k]['regex'] = v
            return sortie
    else:
        champs['regex'] = sature_buffer(champs.get('regex'), i//2)
        return {i: deepcopy(champs) for i in range(1, i + 1)}


def procedure_mokrwaze_1(taille, bdd, grid=None, debug=False):
    """
        Méthode trop poussive, exploration complète de l'arbre
    :param taille: entier ou couple d'entier, déterminant la taille du tableau
    :param bdd: ensemble de chaine
    :param grid: tableau représantant une solution possible
    :param debug: affichage de certains éléments pour voir le déroulement de la fonction
    :return: None, Cette fonction affiche les solutions en print.
    """
    from sys import stderr
    from codecs import open

    sortie = open('phonoku_{}.txt'.format(taille), 'a', 'utf-8')

    def seq_p_i(j):
        """
            Fonction raccourcie qui évite de retaper son contenu.
        :param j: entier
        :return: booléen
        """
        return seq_i if est_pair(j) else seq_p

    length_bdd = len(bdd)
    print(taille)
    length = taille + taille if isinstance(taille, int) else add(*taille)
    length_1 = length + 1
    struc = initialise(length, grid=grid)
    seq_i = range(1, length_1, 2)
    seq_p = range(2, length_1, 2)
    i = 1

    # ligne précédemment utilisée avant l'tilisation des expressions rationnelles
    # struc[i]['posss'] = filter(lambda x: x.startswith(struc.get(i).get('buffer')), bdd)

    if struc.get(i).get('posss') is None:
        regex = struc.get(i).get('regex')
        buffer = sature_buffer(struc.get(i).get('buffer'), taille)
        unif = re.compile(unifie_chaines(regex, buffer))
        if unif is not None:
            struc[i]['posss'] = filter(unif.match, bdd)
        else:
            return

    if debug:
        print('RESTE\tCOURANT\tLENGTH\tI', file=stderr)

    while i <= length:
        # print(i)
        if ((i == 1) and (struc.get(i).get('posss') is None)) or (i < 1):
            break

        answer = next(struc.get(i).get('posss'), None)
        if debug:
            if i == 1:
                length_bdd -= 1
                print('{}\t{}\t{}\t{}'.format(length_bdd, answer, taille, i), file=stderr)
        while answer is not None:
            distribue_soluce(answer, seq_i if est_pair(i) else seq_p, struc)
            if i == length:
                print(*[struc.get(i).get('buffer') for i in seq_i], sep=";", end="\n", file=sortie)
                retire_soluce(seq_i if est_pair(i) else seq_p, struc)
            elif not verifie_posss(seq_p_i(i), struc, bdd):
                if debug:
                    print('{}\t{}\t{}\t{}'.format(length_bdd, answer, taille, i), file=stderr)
                retire_soluce(seq_p_i(i), struc)
            else:
                i += 1
            if struc.get(i).get('posss') is None:
                regex = struc.get(i).get('regex')
                buffer = sature_buffer(struc.get(i).get('buffer'), taille)
                unif = re.compile(unifie_chaines(regex, buffer))
                if unif is not None:
                    struc[i]['posss'] = filter(unif.match, bdd)
                else:
                    break
            answer = next(struc.get(i).get('posss'), None)
        struc[i]['posss'] = None
        if i > 1:
            i -= 1
        retire_soluce(seq_i if est_pair(i) else seq_p, struc)


def verifie_posss(seq, struc, bdd):
    return all(any(re.compile(struc.get(y).get('regex')).match(x) for x in bdd) for y in seq)


def est_pair(x: int): return True if not (x % 2) else False


def main():
    import argparse
    import sqlite3
    import itertools

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'taille',
        type=str
    )

    parser.add_argument(
        'bdd',
        type=sqlite3.connect
    )

    parser.add_argument(
        '-g',
        '--grid',
        type=str
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true'
    )

    args = parser.parse_args('5 ../../Lexiques/Lexique381/Lexique.db -v'.split())

    print(args)

    taille = list(map(int, args.taille.split(',')))
    taille = taille[0] if len(taille) == 1 else taille

    grid = args.grid.split(';') if args.grid else args.grid

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

    cursor = args.bdd.cursor()

    cursor.execute('SELECT DISTINCT phon FROM Lexique WHERE length(phon)=?', (taille,))
    bdd = [neutralise(x, symbs) for x in itertools.chain(*cursor.fetchall())]
    procedure_mokrwaze_1(taille=taille, bdd=bdd, grid=args.grid, debug=args.verbose)

if __name__ == '__main__':
    main()
