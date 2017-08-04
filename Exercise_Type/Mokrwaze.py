# coding: utf-8
import operator
from copy import deepcopy
import re
import numpy
from collections import defaultdict
import itertools
from multiprocessing import Process, Queue, Pool
from sqlite3 import Cursor, connect
import sys
import typing


class InfiniteDict(defaultdict):
    def __init__(self):
        super(InfiniteDict, self).__init__(self.__class__)


def liste_resultats(sequence):
    return itertools.chain(*sequence)


def neutralise(chaine):
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
    return map(
        lambda x: "".join(x),
        map(
            list,
            numpy.transpose(list(map(list, liste)))
        )
    ) if all(len(liste[0]) == len(x) for x in liste) else []

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
        if operator.add(x, y) == i:
            for (k, v) in zip(range(1, i + 1), itertools.chain(*zip(grid, transpose(grid)))):
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
    length = taille + taille if isinstance(taille, int) else operator.add(*taille)
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


def procedure_mokrwaze_2(taille, bdd):
    import itertools
    f = 0
    i = 2
    for k, v in itertools.combinations(bdd.get(taille).get(i), i):
        tmp = list(transpose((k, v)))
        if all(x in bdd.get(taille).get(i) for x in tmp):
            f += 1
    print(f)

            # l, g = tmp
            # i += 1
            # while i <= taille:
            #     print(i, file=sys.stderr)
            #     for mot1 in bdd.get(taille).get(i):
            #         t1 = k + mot1[0]
            #         t2 = v + mot1[1]
            #         if (t1 in bdd.get(taille).get(i)) and (t2 in bdd.get(taille).get(i)):
            #             for mot2 in bdd.get(taille).get(i):
            #                 if mot1[-1] == mot2[-1]:
            #                     t3 = l + mot2[0]
            #                     t4 = g + mot2[1]
            #                     if (t3 in bdd.get(taille).get(i)) and (t4 in bdd.get(taille).get(i)):
            #                         i += 1
            #                         print(k, v, mot2, file=sys.stderr)


def verifie_posss(seq, struc, bdd):
    from subprocess import Popen
    x = Popen('grep ')
    # return all(any(re.compile(struc.get(y).get('regex')).match(x) for x in bdd) for y in seq)


def est_pair(x: int): return True if not (x % 2) else False


def f(bdd, i):
    tmp1 = (z for z in itertools.combinations(bdd, i))
    for x in itertools.combinations(tmp1, 2):
        soluce = list()
        tmp = transpose(x)
        suiv = next(tmp, None)
        while suiv in bdd:
            soluce.append(suiv)
            suiv = next(tmp, None)
        if len(soluce) != i:
            soluce.clear()
        else:
            yield soluce


def g(curseur, column: str, taille: int):
    range_taille = range(1, taille + 1)
    struc = {i: list() for i in range_taille}

    begins = list()
    begs = list()

    for i in range_taille:
        curseur.execute(
            "SELECT DISTINCT substr({column},1,?) FROM lexique where length({column})=?;".format(
                column=column
            ),
            [i, taille]
        )
        struc[i].extend(
            liste_resultats(
                sequence=curseur.fetchall()
            )
        )
    begins = filter(lambda elem: all(x in struc.get(2) for x in transpose(elem)), itertools.combinations(struc.get(2), r=2))
    for l0 in struc.get(2):
        # pour la lihgne qui suit: (l0 <= elem) permet de réduire la liste mais que fait de réduire cette liste ?
        for c0 in filter(lambda elem: (elem[0] == l0[0]) and (l0 <= elem), struc.get(2)):
            for l1 in filter(lambda elem: elem[0] == c0[1], struc.get(2)):
                begs.append([l0, l1] if (l0[1] + l1[1]) in struc.get(2) else None)
    begs = list(filter(None, begs))
    print(list(begins))
    print()
    print(begs)


def colonnes(lignes, debug=False):
    if debug:
        print("lignes:", lignes)
    taille = len(lignes)
    tampon = []
    result = []
    array = []
    for i in range(taille):
            array.append(list(lignes[i]))
    if debug: print("array:", array)
    tampon = map(zip(*array))
    if debug: print("tampon:", tampon)
    for i in range(taille):
            chaine=""
            for element in tampon[i]:
             chaine=chaine+element
            result.append(chaine)
    if debug: print("result:", result)
    return result


def segmente(curseur, column, length):
    """
        A partir d'une base de données, ainsi que d'une colonne précise, on va construire
        une dictionnaire ayant pour clé la taille des sous chaines qui lui correspondront et en valeurs
        les sous chaines correspondantes.
    :param curseur: sqlite3.cursor
    :param column: chaine de caracteres correspondant à une table de la bdd
    :param length: taille maximale de la chaine
    :return: dictionnaire (int, liste)
    """
    struc = dict(zip(range(1, length+1), [None]*length))
    for i in range(1, length+1):
        curseur.execute(
            "SELECT DISTINCT substr({column},1,?) FROM lexique where length({column})=?;".format(
                column=column
            ),
            (i, length)
        )
        struc[i] = list(liste_resultats(curseur.fetchall()))
    return struc


def init_1(bdd: dict):
    """
        Parcours d'une portion de la bdd, pour rechercher les carrés initiaux compatibles
        on yield les lignes.
    :param bdd:
    :return:
    """
    for x, y in itertools.combinations(bdd.get(2), 2):
        if all(z in bdd.get(2) for z in transpose((x, y))):
            yield [x, y]


def extension(args: list, i: int, max: int, bdd: dict, result: open):
    if i == max:
        print(*args, sep=";", file=result)
    else:
        i += 1
        for x, y in itertools.combinations(bdd.get(i), 2):
            if x[-1] == y[-1]:
                xxx = args+[x]
                for j in range(len(y)-1):
                    xxx[j] += y[j]
                calcul = all(z in bdd.get(i) for z in xxx + list(transpose(xxx)))
                if calcul:
                    extension(xxx, i, max, bdd, result)


def ff(bdd, i):
    for elem in itertools.product(bdd.get(i), repeat=i):
        if all(x in bdd.get(i) for x in transpose(elem)):
            print(elem)


def procedure_grille(length: int, inds: tuple, industrie: defaultdict, bdd: Cursor):
    i = 0
    while True:
        if inds == (0, 0):
            id1 = (inds[0] + 1, inds[1])
            id2 = (inds[0], inds[1] + 1)
            [industrie[id1].reception.put(x) for x in industrie[inds].memoire]
            [industrie[id2].reception.put(x) for x in industrie[inds].memoire]
        else:
            print("lecture:", inds, file=sys.stderr)
            grille = industrie[inds].reception.get()
            print(type(grille))
            for grid in industrie.get(inds).memooire:
                calcul = add(grille, grid)
                if calcul is not None:
                    id1 = (inds[0] + 1, inds[1])
                    id2 = (inds[0], inds[1] + 1)
                    industrie[id1].reception.put(calcul)
                    industrie[id2].reception.put(calcul)
        if inds == (length, length):
            grille = industrie[inds].reception.get()
            print(grille)
            i += 1
            cmd = "INSERT INTO Mokrwaze(mo_num, ligne) VALUES (?, ?)"
            bdd.executemany(cmd, ["".join(x) for x in grille])


def initialise_memoire(i: int, k: int, sequences: list):
    from itertools import combinations
    from numpy import chararray
    from collections import defaultdict

    struc = defaultdict(Zone)
    if k == 1:
        for r in range(i):
            grid = chararray((i, i))
            for (x, y) in combinations(iterable=sequences, r=2):
                grid[r] = list(x)
                grid[:, r] = list(y)
                struc[(r, r)].memoire = grid
    elif k == 2:
        for r in range(i):
            for s in range(i):
                grid = chararray((i, i))
                for v, w in combinations(iterable=sequences, r=2):
                    grid[r] = list(v)
                    grid[:, s] = list(w)
                    struc[(r, s)].memoire = grid
    return struc


class Zone(object):
    def __init__(self, outputs, func=lambda x, y: (x,y)):
        self.envoi = Queue()
        self.reception = Queue()
        self.traitement = Queue()
        self.memoire = set()
        self.process_traitement = Process(target=self.procedure_traitement, args=(func,))
        self.process_envoi = Process(target=self.procedure_envoi, args=(outputs,))
        self.process_reception = Process(target=self.procedure_reception)
        self.process_traitement.start()
        self.process_envoi.start()
        self.process_reception.start()
        self.process_traitement.join()
        self.process_envoi.join()
        self.process_reception.join()

    def procedure_envoi(self, outputs):
        while True:
            if not self.envoi.empty():
                tmp = self.envoi.get()
                for zone in outputs:
                    zone.reception.put(tmp)

    def procedure_reception(self):
        while True:
            if not self.reception.empty():
                tmp = self.reception.get()
                self.traitement.put(tmp)

    def procedure_traitement(self, func):
        while True:
            if not self.traitement.empty():
                elem = func(self.traitement.get())
                if elem is not None:
                    self.envoi.put(elem)


    def initialise(self, n, infos):
        zones = list()
        zones.append(Zone(infos, (1,1), lambda x, y: x.add(y)))


class Grid(object):
    def __init__(self, shape, itemsize=1, unicode=False, buffer=None, offset=0,
                 strides=None, order='C', jonction=(), symmetric=False):
        self.grille = numpy.chararray(
            shape,
            itemsize=itemsize,
            unicode=unicode,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order
        )
        self.jonction = jonction
        self.symmetric = symmetric

    def __repr__(self): return str(self.grille)

    def __str__(self): return repr(self)

    def is_symmetric(self):
        return (self.grille[self.jonction[0]] == self.grille[:, self.jonction[1]]).all()

    def naive_add(self, other) -> typing.Union[numpy.chararray, None]:
        g3 = numpy.chararray(shape=self.grille.shape)
        for i in self.grille:
            for j in self.grille:
                if self.grille[i, j] == other.grille[i, j]:
                    g3[i][j] = self.grille[i][j]
                elif (self.grille[i][j] + other.grille[i][j]) == (self.grille[i][j] or other.grille[i][j]):
                    g3[i][j] = (self.grille[i][j] or other.grille[i][j])
                else:
                    return None
        return g3

    def add_optimise(self, other):
        """
            le i passé en paramètre permet de restreindre drastiquement le nombre de comparaison.
            Si i > 0 alors on comparé (g1[], g1[])
        :param g1:
        :param g2:
        :param i:
        :return:
        """
        g3 = self.grille.copy()
        (x, y) = other.jonction
        verif = ((x-1, x), (y, y-1))
        if (self.grille[verif[0]] == other.grille[verif[0]]) and (self.grille[verif[1]] == other.grille[verif[1]]):
            for i in range(x, g3.shape[0]):
                for j in range(y, g3.shape[0]):
                    g3[i, j] = other.grille[i, j]



def genere_structure(corpus: list, n: int, i: int):
    assert all(len(x) == n for x in corpus)
    assert i < n

    def f(*args):
        (mot1, mot2, i) = args
        array = Grid(shape=(n, n), unicode=True)
        array.jonction = (i, i)
        array.symmetric = mot1 == mot2
        array.grille[i] = list(mot1)
        array.grille[:, i] = list(mot2)
        return array

    # struc = dict()

    # struc[i] = dict()
    dd = list()
    for k, g in itertools.groupby(corpus, lambda x: x[i]):
        dd.extend(g)
    return i, ((i+1, i), (i, i+1)) if i < n else None, itertools.starmap(f, map(lambda x: x + (i,), itertools.product(dd, repeat=2)))


def main():
    import argparse
    import sqlite3
    from codecs import open
    from collections import defaultdict

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

    args = parser.parse_args('3 ../../Lexiques/Lexique381/Lexique.db -v'.split())

    print(args)

    taille = list(map(int, args.taille.split(',')))
    taille = taille[0] if len(taille) == 1 else taille

    grid = args.grid.split(';') if args.grid else args.grid

    args.bdd.create_function('neutralise', 1, neutralise)
    cursor = args.bdd.cursor()

    cursor.execute('SELECT DISTINCT neutralise(phon) FROM Lexique WHERE length(phon)=?', (taille,))
    bdd = segmente(cursor, 'phon', taille)
    for element in init_1(bdd):
        print(element)
        extension(element, 2, 3, bdd, open(filename='phonoku_{}.txt'.format(taille), mode='w', encoding='utf-8'))


    # print(*bdd, file=open('bdd_{}'.format(taille), 'w', 'utf-8'))

    # procedure_mokrwaze_1(taille=taille, bdd=bdd, grid=args.grid, debug=args.verbose)


def main2():
    with connect('mokrwaze.db') as bdd:
        cursor = bdd.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS Mokrwaze(
    id INTEGER PRIMARY KEY,
    mo_num INTEGER,
    ligne TEXT
)""")
    length = 5
    sequences = list()
    k = 1
    dico_process = defaultdict()
    industrie = initialise_memoire(i=length, k=k, sequences=sequences)
    if k == 1:
        for i in range(length):
            print(i, i)
            dico_process[(i, i)] = Process(target=procedure_grille, args=(length, (i, i), industrie, cursor))
            dico_process[(i, i)].start()
            # dico_process[(i, i)].join()
        exit()
    elif k == 2:
        for i in range(length):
            for j in range(length):
                dico_process[(i, j)] = Process(target=procedure_grille, args=(length, (i, j), industrie, cursor))
                dico_process[(i, i)].start()
                dico_process[(i, i)].join()


def main3():
    n = 3
    with connect('Lexique.db') as corpus:
        corpus_curs = corpus.cursor()
    lexique = list(itertools.chain(*corpus_curs.execute('select distinct ortho from lexique where length(ortho)=?', (n,)).fetchall()))
    tmp = genere_structure(lexique, n, 1)
    print(tmp, next(tmp[-1]))
    print(tmp, next(tmp[-1]))
    # task_queue = Queue()

    # for x in tmp:
    #     task_queue.put(x)


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)


if __name__ == '__main__':
    main3()