# coding: utf-8
from chest import Chest
from multiprocessing import Process, Pool, Queue, Manager
from collections import defaultdict, deque
# import re
from Exercise_Type.Powerset import powerset_tostring
from Exercise_Type.combinations import combinations
from Exercise_Type.Powerset import Variable
from Exercise_Type.tfidf import tf_idf
from sys import stderr
from heapdict import heapdict
from itertools import chain, product, count
import sqlite3
import string
import functools
# from codecs import open
import psycopg2
# from psycopg2 import sql
from codecs import open
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class Point(object):
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return str(self) == str(other)


class OptimString:
    def __init__(self, seq: list, point: Point, pointee=None, control=None):
        self.__point = point
        self.__data = seq
        self.__data_pointe = list(seq) if not pointee else pointee
        self.__control = defaultdict(deque) if not control else control
        if not self.__control:
            self.__remplir_control(seq)
        tmp = list(filter(lambda x: isinstance(x, str), self.__data_pointe))
        self.__first = tmp[0] if tmp else ''

    @property
    def first(self):
        return self.__first

    @property
    def get_point(self):
        return self.__point

    @property
    def data_pointe(self):
        return self.__data_pointe

    @data_pointe.setter
    def data_pointe(self, value):
        pass

    @property
    def data(self):
        return self.__data

    @property
    def control(self):
        return self.__control

    def etendre(self):
        l = list()
        n_p = None
        for x in self.data_pointe:
            if not isinstance(x, Point):
                if n_p is not None:
                    l.append(n_p)
                    n_p = None
                l.append(x)
            else:
                if n_p is None:
                    n_p = x
                else:
                    n_p = Point('.+')
        if n_p is not None:
            l.append(n_p)
        return OptimString(point=Point('.+'), seq=l)

    def __remplir_control(self, seq): list(
        map(lambda x: self.control[x[0]].append(x[1]), self.__inverse_enumerate(seq=seq))
    )

    def trier_control(self): return heapdict(self.control)

    @staticmethod
    def __inverse_enumerate(seq):
        """
            On inverse clé,valeur de enumerate(sequence).
        :return:
        """
        return ((t, y) for y, t in enumerate(seq))

    def add_point(self):
        data_pointe = deepcopy(self.data_pointe)
        control = deepcopy(self.control)

        if self.get_point not in control:
            data_pointe[0] = self.get_point
            control[self.get_point].append(0)
        else:
            if isinstance(data_pointe[-1], Point):
                return OptimString(point=self.__point, seq=self.data, pointee=data_pointe, control=control)

            pt = control.get(self.get_point)[-1]
            data_pointe[pt+1] = self.get_point
            control.get(self.get_point).append(pt+1)

        return OptimString(point=self.__point, seq=self.data, pointee=data_pointe, control=control)

    def deplace_point(self):
        data_pointe = deepcopy(self.data_pointe)
        control = deepcopy(self.control)

        if self.get_point not in control:
            return self

        pt = control.get(self.get_point)[-1]
        if pt == len(self.__data)-1:
            return OptimString(point=self.__point, seq=self.data, pointee=data_pointe, control=control)
        else:
            control[self.get_point][-1] += 1
            data_pointe[pt] = self.__data[pt]
            data_pointe[pt+1] = self.get_point
        return OptimString(point=self.__point, seq=self.data, pointee=data_pointe, control=control)

    def __repr__(self):
        return "".join(map(lambda x: str(x), self.data_pointe))

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.data_pointe == other.data_pointe

    def __hash__(self):
        return 1


def escape_char(x):
    return "\\"+x if ((str(x) in string.punctuation) and isinstance(x, str)) else str(x)


def generate_regex(seq: OptimString, classes, cursor: sqlite3.Cursor):
    """
        powerset appliqué selon la logique d'un parcours en largeur d'un graphe.
    :param seq: chaine de caractère OptimString plus optimisé que str
    :param classes: liste de classes
    :param cursor: cursuer d'une db sqlite3
    :return: None
    """
    doublon = None

    add_value_cmd = """INSERT INTO Examples(corpus_id,class_id,feature_id)
    SELECT Corpus.id,Classes.id,Features.id
    FROM Corpus,Classes,Features
    WHERE Corpus.sequence=?
    AND Classes.classe=?
    AND Features.feature=?;
    """
    add_value_cmd_2 = "INSERT OR IGNORE INTO Examples(corpus_id,class_id,feature_id) VALUES (?,?,?);"
    add_classes_cmd = "INSERT OR IGNORE INTO Classes(classe) VALUES (?);"
    add_features_cmd = "INSERT OR IGNORE INTO Features(feature) VALUES (?);"
    add_sequence_cmd = "INSERT OR IGNORE INTO Corpus(sequence) VALUES (?);"
    add_genealogie_cmd = "INSERT OR IGNORE INTO Genealogie(pere_id, descendant_id) VALUES (?,?);"
    select_feature = "SELECT id FROM Features WHERE feature IN ({});"
    select_classe = "SELECT id FROM Classes WHERE classe IN ({});"
    select_descendant = "SELECT descendant_id FROM genealogie WHERE pere_id IN ({});"

    seq_id = cursor.execute(select_feature.format(",".join(["?"]*len((str(seq),)))), (str(seq),)).fetchone()
    str_seq = str(seq)

    file = deque()

    file.appendleft(seq)

    while file:
        current = file.pop()
        str_current = str(current.etendre())

        cursor.execute(add_features_cmd, (str_current,))
        # vérification de la présence de current dans la database
        cursor.execute(select_feature.format(",".join(["?"]*len((str_current,)))), (str_current,))
        feat = cursor.fetchone()
        cursor.execute(select_descendant.format(",".join(["?"]*len(feat))), feat)
        descendant_ids = list(filter(lambda x: x != ('NULL',), cursor.fetchall()))
        if descendant_ids:
            classe_ids = cursor.execute(select_classe.format(",".join(["?"]*len(classes))), classes).fetchall()
            cursor.executemany(
                add_value_cmd_2, map(lambda x: (x[0] + x[1] + x[2]), product([seq_id], classe_ids, descendant_ids))
            )
        else:
            # mettre à jour features
            cursor.execute(add_features_cmd, (str_current,))
            if str_seq != str_current:
                tmp = (str_seq, str_current if not all(x == '.' for x in str_current) else 'NULL')
                if tmp != doublon:
                    if isinstance(current.data_pointe[-1], Point):
                        # mettre à jour généalogie
                        cursor.execute(
                            add_genealogie_cmd,
                            cursor.execute(
                                select_feature.format(",".join(["?"]*len((str_current,)))),
                                (str_current,)
                            ).fetchone() + ("NULL",)
                        )
                    else:
                        # mettre à jour généalogie
                        cursor.execute(add_genealogie_cmd,
                                       seq_id + cursor.execute(select_feature.format(",".join(["?"]*len((str_current,)))), (str_current,)).fetchone()
                                       )
                    # mettre à jour les exemples
                    cursor.executemany(add_value_cmd, map(lambda x: (str_seq, x, str_current), classes))
                doublon = tmp
            if not isinstance(current.data_pointe[-1], Point):
                old = str_current
                current = current.add_point()
                str_current = str(current.etendre())
                tmp = (old, str_current if not all(x == '.' for x in str_current) else 'NULL')
                if tmp != doublon:
                    # mettre à jour features
                    cursor.execute(add_features_cmd, (str_current,))
                    # mettre à jour les exemples
                    cursor.executemany(add_value_cmd, map(lambda x: (str_seq, x, str_current), classes))
                    # mettre à jour généalogie
                    cursor.execute(add_genealogie_cmd, seq_id + cursor.execute(select_feature.format(",".join(["?"]*len((str_current,)))), (str_current,)).fetchone())
                if not all(x == '.' for x in str_current):
                    file.appendleft(current)
                for _ in range(current.control.get(current.get_point)[-1]+1, len(current.data)):
                    current = current.deplace_point()
                    file.appendleft(current)
                doublon = tmp


def skip_duplicates(iterable, key=lambda x: x):
    """
        http://sametmax.com/saffranchir-des-doublons-dun-iterable-en-python/
    :param iterable:
    :param key:
    :return:
    """
    # on va mettre l’empreinte unique de chaque élément dans ce set
    fingerprints = set()

    for x in iterable:
        # chaque élement voit son emprunte calculée. Par défaut l’empreinte
        # est l'élément lui même, ce qui fait qu'il n'y a pas besoin de
        # spécifier 'key' pour des primitives comme les ints ou les strings.
        fingerprint = key(x)

        # On vérifie que l'empreinte est dans la liste des empreintes  des
        # éléments précédents. Si ce n'est pas le cas, on yield l'élément, et on
        # rajoute sont empreinte ans la liste de ceux trouvés, donc il ne sera
        # pas yieldé si on ne le yieldera pas une seconde fois si on le
        # rencontre à nouveau
        if fingerprint not in fingerprints:
            yield x
            fingerprints.add(fingerprint)


def remove_duplicates(lst, equals=lambda x, y: x == y):
    """
        http://sametmax.com/saffranchir-des-doublons-dun-iterable-en-python/
    :param lst:
    :param equals:
    :return:
    """
    if not isinstance(lst, list):
        raise TypeError('This function works only with lists.')
    i1 = 0
    l = (len(lst) - 1)
    while i1 < l:
        elem = lst[i1]
        i2 = i1 + 1
        while i2 <= l:
            if equals(elem, lst[i2]):
                del lst[i2]
                l -= 1
            i2 += 1
        i1 += 1
    return lst


class Perceptron(object):
    """
        Cette classe Perceptron est une fusion entre celle vue en cours extraite du tp4 et le code du perceptron moyenné
        fourni pour réaliser ce projet.

    """

    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))

    def __perceptron_update(self, label: str, feature: str, quantity: float):
        from random import random
        if self.weights.get(label).get(feature):
            self.weights[label][feature] += quantity
        else:
            self.weights[label][feature] += random()
            self.weights[label][feature] += quantity

    def update(self, true_label: str, guessed_label: str, features: defaultdict):
        if true_label == guessed_label:
            return
        for k, v in features:
            self.__perceptron_update(k, true_label, +v)
            self.__perceptron_update(k, guessed_label, -v)

    def score(self, features, labels=None):
        if labels is None:
            labels = self.weights.keys()

        scores = defaultdict(float)
        for c in labels:
            scores[c] = self.decision_function(features, c)

        return scores

    def predict(self, features):
        # scores = self.score(features=features, labels=possible_labels)
        classe = str()
        value = float()
        for c in self.weights.keys():
            calcul = self.decision_function(features, c)
            if calcul > value:
                classe = c
                value = calcul
        return classe
        # return max(scores, key=lambda label: (scores[label], label))

    def normalize(self, word):
        """
            Remplace certains types de tokens par des pseudos-mots

        :param word: string
        :return: string
        """
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        elif word == "X":
            return "NOUN"
        else:
            return word.lower()

    def train(self, train_data, test_data, iter_max):
        import random, sys

        for i in range(iter_max):
            for (vecteur, classe) in train_data:
                self.update(classe, self.predict(features=vecteur), vecteur)
            print("ACCURACY\tI\tTRAIN\tTEST", file=sys.stderr)
            print("\t" + "\t".join((i, self.evaluate(train_data), self.evaluate(test_data))), file=sys.stderr)
            random.shuffle(train_data)

    def evaluate(self, examples):
        acc = 0.0
        total = 0.0
        for (vecteur, classe) in examples:
            pred = self.predict(vecteur)
            if classe == pred:
                acc += 1.
            total += 1
        return acc / total

    def decision_function(self, features, label):
        from functools import reduce
        from operator import mul, add
        return reduce(
            lambda x, y: add(x, y),
            map(
                lambda f: mul(
                    features.get(f),
                    self.weights.get(label).get(f)
                ),
                features
            )
        )


class PerceptronSequentiel(Perceptron):
    def __init__(self):
        super(PerceptronSequentiel, self).__init__()
        self.weights = defaultdict(lambda: defaultdict(float))
        # valeur accumulée de la paire classe/feature
        self._cached = defaultdict(lambda: defaultdict(float))
        # nombre d'occurence déjà rencontrée
        self.n_updates = 0.0
        # étiquettes de début de séquence
        self.START = ['-START-', '-START2-']
        # étiquettes de fin de séquence
        self.END = ['-END-', '-END2-']


def update(true_label, guessed_label, features, weights, n_updates=None):
    if true_label == guessed_label:
        return

    for f, v in features.items():
        if true_label in weights:
            weights[true_label][f] += v
            weights[guessed_label][f] -= v


def score(features, weights):
    return dict(map(lambda c: (c, dot_product(features, c, weights)), weights.keys()))


def dot_product(features, label, weights):
    """
        Calcul de la fonction décision

    :param features: dictionnaire
    :param label: string
    :return: float
    """
    from functools import reduce
    from operator import add, mul

    return reduce(add, map(lambda f: mul(features[f], weights[label][f]), features))


def predict(features, weights):
    classe = str()
    value = float()
    for c in weights:
        calcul = dot_product(
            features=features,
            label=c,
            weights=weights
        )
        if calcul > value:
            classe = c
            value = calcul
    return classe


def procedure_perceptron(max, corpus, examples, examples_queue, sequences_queue):
    for i in range(max):
        # shuffle(corpus)
        for (classe, sequence) in corpus:
            if sequence in examples:
                print("perceptron: ", i, sequence, "example", file=stderr)
                examples_queue.put((i, classe, examples.get(sequence)))
            else:
                print("perceptron: ", i, sequence, classe, "powerset", file=stderr)
                sequences_queue.put((i, classe, sequence))


def procedure_example(examples_queue, test_queue, weights):
    while True:
        j = int()
        entree = examples_queue.get()
        if entree:
            (i, classe, vecteur) = entree
            if i != j:
                [examples_queue.put(x) for x in examples.keys()]
                # test_queue.put((i, weights))
                j = i
            else:
                print('maj', file=stderr)
                update(
                    true_label=classe,
                    guessed_label=predict(
                        features=vecteur,
                        weights=weights
                    ),
                    features=vecteur,
                    weights=weights
                )


def procedure_test(test_set, weights):
    """
        Si une classe du test_set n'est pas dans le train_set, l'example est ignoré
        si une dimension n'est pas dans les features, ont la classifie sous le nom ##OUTLAW##
    :param test_set:
    :return:
    """
    print("I", "ACC", sep='\t', end='\t\n', file=stderr)
    while True:

        acc = 0.0
        total = 0.0
        # prise en compte du problème des classes inconnues
        for classe, vecteur in filter(lambda x: x[0] in weights, test_set):
            pred = predict(vecteur, weights)
            if classe == pred:
                acc += 1.
            total += 1
        print("Tour {i}; ACC: {prec}".format(i=i, prec=acc / total))


def procedure_powerset(sequences_queue: Queue, examples_queue: Queue, examples: dict, corpus: list):
    while True:
        (i, classe, sequence) = sequences_queue.get()
        print(sequence)
        with Pool(10) as proc:
            args = map(lambda x: (sequence, x, Variable('.'), Variable('.+'), True), combinations(len(sequence)))
            for regex in proc.map(powerset_tostring, list(args)):
                examples[sequence] = dict()
                calcul = tf_idf(regex, sequence, list(map(lambda x: x[1], corpus)))
                examples[sequence][regex] = calcul
            examples_queue.put((i, classe, examples.get(sequence)))


def maj(sequences, classes, features, data, cursor: sqlite3.Cursor, debug=False):
    add_value_cmd = "INSERT INTO Examples(corpus_id,class_id,feature_id,value) VALUES (?,?,?,0.0)"
    select_ids = '''SELECT Corpus.id,Classes.id,Features.id
FROM Corpus,Classes,Features
WHERE Corpus.sequence=? AND Classes.classe=? AND Features.feature=?'''
    add_classes_cmd = "INSERT OR IGNORE INTO Classes(classe) VALUES (?)"
    add_features_cmd = "INSERT OR IGNORE INTO Features(feature) VALUES (?)"
    add_sequence_cmd = "INSERT OR IGNORE INTO Corpus(sequence) VALUES (?)"
    cursor.execute('BEGIN')
    to_tuple = lambda x: (x,)
    to_tuple_seq = lambda t: map(to_tuple, iter(t))

    # màj des séquences du corpus
    cursor.executemany(add_sequence_cmd, to_tuple_seq(sequences))
    sequences.clear()

    # màj des classes
    cursor.executemany(add_classes_cmd, to_tuple_seq(classes))
    classes.clear()

    # màj des features
    cursor.executemany(add_features_cmd, to_tuple_seq(features))
    features.clear()

    # màj des examples
    cursor.executemany(
        add_value_cmd,
        list(
            map(
                lambda x: cursor.execute(select_ids, x).fetchone(),
                data
            )
        )
    )
    if debug:
        print("màj effectuée", file=stderr)


def bfs(descendant: OptimString, func, cursor, pere=None):
    """
        Breadth First Search
    :return:
    """
    fifo = deque()

    func(pere, descendant, fifo, cursor)
    while fifo:
        pere = fifo.pop()
        cible = pere.add_point().etendre()
        if not func(pere, cible, fifo, cursor):
            for x in range(len(cible)):
                func(cible, cible.deplace_point().etendre(), fifo, cursor)


def func(new: OptimString, fifo: deque, curseur: sqlite3.Cursor, classes, old: OptimString=None) -> bool:
    cmds = {
        "select": "SELECT {columns} FROM {tables} WHERE {column} IN ({values}) ",
        "insert": "INSERT OR IGNORE INTO {table}({columns}) VALUES ({values})",
        "insert_select": "INSERT OR IGNORE INTO {table}({columns}) {SELECT}"
    }

    def select_stmnt(columns, tables, column, values):
        return cmds.get('select').format(",".join(columns), ",".join(tables), column, ",".join(["?"]*len(values)))

    def insert_stmnt(table, columns, values):
        return cmds.get('insert').format(table, ",".join(columns), ",".join(["?"]*len(values)))

    def insert_select_stmnt(table, columns, select):
        return cmds.get('insert').format(table, ",".join(columns), select)

    existence = curseur.execute(select_stmnt(('id',), ('features',), ('feature',), (str(new),)), (str(new),)).fetchone()
    if existence:
        descendants = curseur.execute(select_stmnt(('descendant_id',), ('Genealogie',), ('pere_id',), existence), existence).fetchall()
        if descendants:
            seq_id = curseur.execute(select_stmnt(('id',), ('features',), ('feature',), (str(new),)), (str(new),))
            classe_ids = curseur.execute(select_stmnt(('id',), ('classes',), ('classe',), classes), classes).fetchall()
            for args in product(seq_id, classe_ids, descendants):
                args = functools.reduce(lambda x,y: x+y, args)
                curseur.execute(insert_stmnt('Examples', ('corpus_id', 'classe_id', 'feature_id'), args), args)
        return True
    else:
        curseur.execute(insert_stmnt('Features', ('feature',), (str(new),)), (str(new),))
        values = functools.reduce(
            lambda x, y: x + y,
            curseur.execute(
                select_stmnt(('id',), ('Features',), 'feature', (new.data, str(new))),
                (new.data, str(new))
            ).fetchall()
        )
        curseur.execute(insert_stmnt('Genealogie', ('pere_id', 'descendant_id'), values), values)
        values = functools.reduce(
            lambda x, y: x + y,
            curseur.execute(
                select_stmnt(('id',), ('Features',), 'feature', (str(old), str(new))),
                (str(old), str(new))
            ).fetchall()
        )
        curseur.execute(insert_stmnt('Genealogie', ('pere_id', 'descendant_id'), values), values)
        if isinstance(new.data_pointe[-1], Point):
            values = functools.reduce(
                lambda x, y: x + y,
                curseur.execute(
                    select_stmnt(('id',), ('Features',), 'feature', (str(new),)),
                    (str(new),)
                ).fetchall()
            )
            curseur.execute(insert_stmnt('Genealogie', ('pere_id',), values), values)
        fifo.appendleft(new)
        return False


def alpha(i: int, j: int, classes, regex: str, sequence: str, seuil: int, bdd, bdd_cursor, statements, debug: bool=False):
    add_value_cmd = '''INSERT INTO Examples(corpus_id,class_id,feature_id)
            SELECT d.id,Classes.id,Features.id
            FROM Features as d,Classes,Features
            WHERE d.feature=%s AND
            Classes.classe=%s AND
            Features.feature=%s ON CONFLICT DO NOTHING;'''
    add_value_cmd_bis = '''INSERT INTO Examples(corpus_id,class_id,feature_id)
                SELECT d.id,Classes.id,Features.id
                FROM Features as d,Classes,Features
                WHERE d.feature='%s' AND
                Classes.classe='%s' AND
                Features.feature='%s' ON CONFLICT DO NOTHING;'''
    add_classes_cmd = 'INSERT INTO Classes(classe) VALUES (%s) ON CONFLICT DO NOTHING;'
    add_classes_cmd_bis = 'INSERT INTO Classes(classe) VALUES (\'%s\') ON CONFLICT DO NOTHING;'
    add_features_cmd = 'INSERT INTO Features(feature) VALUES (\'%s\') ON CONFLICT DO NOTHING;'
    add_features_cmd_bis = 'INSERT INTO Features(feature) VALUES (\'%s\') ON CONFLICT DO NOTHING;'

    if i == seuil:
        statements.append("COMMIT;")
        bdd_cursor.execute("\n".join(statements))
        statements.clear()
        if debug:
            print("alpha: ", j, i, classes, regex, file=stderr)
        statements.append("BEGIN;")
    [statements.append(add_classes_cmd_bis % (x,)) for x in classes]
    statements.append(add_features_cmd_bis % (regex,))
    [statements.append(add_value_cmd_bis % x) for x in map(lambda x: (sequence, x, regex), classes)]


def procedure_corpus_parallel(corpus: list, bdd, bdd_cursor: sqlite3.Cursor, seuil: int=100, debug: bool=False) -> None:
    i = 0
    statements = list()
    add_value_cmd = '''INSERT INTO Examples(corpus_id,class_id,feature_id)
        SELECT d.id,Classes.id,Features.id
        FROM Features as d,Classes,Features
        WHERE d.feature=%s AND
        Classes.classe=%s AND
        Features.feature=%s ON CONFLICT DO NOTHING;'''
    add_value_cmd_bis = '''INSERT INTO Examples(corpus_id,class_id,feature_id)
            SELECT d.id,Classes.id,Features.id
            FROM Features as d,Classes,Features
            WHERE d.feature='%s' AND
            Classes.classe='%s' AND
            Features.feature='%s' ON CONFLICT DO NOTHING;'''
    add_classes_cmd = 'INSERT INTO Classes(classe) VALUES (%s) ON CONFLICT DO NOTHING;'
    add_classes_cmd_bis = 'INSERT INTO Classes(classe) VALUES (\'%s\') ON CONFLICT DO NOTHING;'
    add_features_cmd = 'INSERT INTO Features(feature) VALUES (\'%s\') ON CONFLICT DO NOTHING;'
    add_features_cmd_bis = 'INSERT INTO Features(feature) VALUES (\'%s\') ON CONFLICT DO NOTHING;'
    # bdd_cursor.execute("BEGIN TRANSACTION;")
    statements.append("BEGIN;")
    with Pool(10) as proc:
        for (j, (classes, sequence)) in corpus:
            sequence = sequence.replace("'", "''")
            if debug:
                print("procedure: ", len(corpus) - j, i, sequence, file=stderr)
            if len(sequence) == 1:
                [statements.append(add_classes_cmd_bis % (x,)) for x in classes]
                statements.append(add_features_cmd_bis % (sequence,))
                [statements.append(add_value_cmd_bis % x) for x in map(lambda x: (sequence, x, sequence), classes)]
                i += 1
            else:
                args = map(lambda x: (sequence, x, Variable('.'), Variable('..*'), True), combinations(len(sequence)))
                for regex in proc.imap_unordered(powerset_tostring, args):
                    regex = regex.replace("'", "''")
                    alpha(
                        i=i,
                        j=j,
                        classes=classes,
                        regex=regex,
                        sequence=sequence,
                        seuil=seuil,
                        bdd=bdd,
                        bdd_cursor=bdd_cursor,
                        statements=statements,
                        debug=True
                    )
                    if i == seuil:
                        i = 0
                    i += 1


def procedure_corpus2(corpus, cursor: sqlite3.Cursor, debug: bool=False):
    for (j, (classes, sequence)) in corpus:
        if debug:
            print("procedure: ", len(corpus) - j, sequence, file=stderr)
        add_feature_cmd = "INSERT OR IGNORE INTO Features(feature) VALUES (?);"
        add_classes_cmd = "INSERT OR IGNORE INTO Classes(classe) VALUES (?);"
        cursor.execute('BEGIN TRANSACTION;')
        cursor.execute(add_feature_cmd, (sequence,))
        cursor.executemany(add_classes_cmd, zip(classes))
        generate_regex(
            OptimString(sequence, Point('.'), pointee=None, control=None),
            classes=classes,
            cursor=cursor
        )


def procedure_tfidf(queue: Queue, bdd, cursor, corpus, debug=False):
    add_value_cmd = "INSERT INTO Examples(example_id,class_id,feature_id,value) VALUES (?,?,?,?)"
    add_value_cmd = "UPDATE Examples SET value=? WHERE example_id=? AND class_id=? AND feature_id=?"
    select_ids = 'SELECT Classes.id,Features.id from Classes,Features WHERE Classes.classe=? and Features.feature=?'

    while True:
        if queue:
            (ex_num, classe, feature, sequence) = queue.get()
            if debug:
                print("******************", ex_num, classe, feature, sequence, file=stderr)
            cursor.execute(select_ids, (classe, feature))
            ids = cursor.fetchone()
            cursor.execute(
                add_value_cmd,
                (tf_idf(feature, sequence, corpus), ex_num, *ids)
            )
        bdd.commit()


def create_tables(cursor):
    examples = """CREATE TABLE IF NOT EXISTS Examples (
    id SERIAL PRIMARY KEY,
    corpus_id INTEGER,
    class_id INTEGER,
    feature_id INTEGER,
    value REAL,
    foreign key (class_id) REFERENCES Classes(id),
    foreign key (feature_id) REFERENCES Features(id),
    foreign key (corpus_id) REFERENCES Features(id),
    unique (corpus_id,class_id,feature_id)
) TABLESPACE Perceptron"""
    classes = """CREATE TABLE IF NOT EXISTS Classes (
    id SERIAL PRIMARY KEY,
    classe TEXT UNIQUE
) TABLESPACE Perceptron"""
    features = """CREATE TABLE IF NOT EXISTS Features (
    id SERIAL PRIMARY KEY,
    feature TEXT UNIQUE
) TABLESPACE Perceptron"""
    genealogie = """CREATE TABLE IF NOT EXISTS Genealogie (
    id SERIAL PRIMARY KEY,
    pere_id INTEGER NOT NULL,
    descendant_id INTEGER,
    FOREIGN KEY (pere_id) REFERENCES Features(id),
    FOREIGN KEY (descendant_id) REFERENCES Features(id),
    UNIQUE (pere_id,descendant_id)
) TABLESPACE Perceptron"""
    cursor.execute(classes)
    cursor.execute(features)
    cursor.execute(examples)
    cursor.execute(genealogie)


def deepcopy(d: object):
    from _pickle import dumps, loads
    return loads(dumps(d))


def verify_format(q):
    if q == "train":
        return [
            ("ADV", "vite"),
            ("VERB", "mangeasse"),
            ("ADJ", "rapide"),
            ("ADJ", "lent"),
            ("CONJ", "or"),
            ("PREP", "à"),
            ("VERB", "vivera"),
            ("ADV", "rapidement"),
            ("ADJ", "frais"),
            ("PREP", "de")
        ]
    elif q == 'test':
        return [
            ("ONO", "oh"),
            ("VERB", "lava"),
            ("ADV", "carrément"),
            ("ADJ", "qualifiées"),
            ("PREP", "avant"),
            ("CONJ", "que"),
            ("VERB", "rangeant"),
            ("ADV", "pertinemment"),
            ("PTCP", "qualifié"),
            ("PREP", "de")
        ]
    else:
        raise Exception("on ne peut choisir qu'entre train ou test")


def main():
    from multiprocessing import Queue, Process
    # import argparse

    # parser = argparse.ArgumentParser(prog="perceptron", description="perceptron à execution asynchrone")
    #
    # parser.add_argument(
    #     "corpus",
    #     type=argparse.FileType(
    #         mode='r',
    #         encoding='utf-8'
    #     ),
    #     help="ensemble de couple (objet, classe)"
    # )
    #
    # parser.add_argument(
    #     "iterMax",
    #     type=int,
    #     default=5,
    #     help="nombre de tours que le classifieur va tourner"
    # )
    #
    # parser.add_argument(
    #     "-s",
    #     "--sequential",
    #     action='store_true',
    #     default=False
    # )
    #
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     help="méthode affichant étape par étape ce qui se passe."
    # )

    # args = parser.parse_args()

    # corpus = args.corpus

    corpus = [
        ("carrément", "ADV"),
        ("constitutition", "ADV"),
        ("véritablement", "ADV"),
        ("camarade", "NOUN"),
        ("véritable", "ADJ"),
        ("cartable", "NOUN"),
        ("plage", "NOUN"),
        ("vite", "ADJ")
    ]

    examples = Chest(path='examples')
    weights = Chest(path='weights')
    iterMax = 5
    i = 0

    # counting_queue = Queue()
    # examples_queue = Queue()


    # phase d'initialisation

    # sequence_queue, counting_queue, examples_queue, examples
    sequence_queue = Queue()
    list(map(lambda y: sequence_queue.put(y), corpus))
    process_powerset = Process(target=traiter_sequence,
                               args=(sequence_queue, examples, list(map(lambda x: x[0], corpus))), name="powerset")
    process_powerset.start()
    process_powerset.join()
    # process_counting = Process(target=traiter_comptage, args=(counting_queue, examples), name="tfidf")
    # process_example = Process(target=traiter_example, args=(examples_queue, examples, i, iterMax, weights), name="example")
    # ex_queue, examples, i, max, weights

    # process_counting.start()
    # process_example.start()


    # process_counting.join()
    # process_example.join()


def main2():
    from _pickle import dumps, loads
    from collections import defaultdict

    print('debut')
    dico = dict()
    for i in range(50):
        for j in range(1000000):
            dico[i] = dict()
            dico[i][j] = (i, j)
    print('fin')
    f = loads(dumps(dico))
    print(f == dico)
    print('copie terminée')


def main3():
    perceptron_bdd = '/Volumes/RESEARCH/Research/Perceptron_2.db'
    lexicon_bdd = 'Lexique.db'
    with sqlite3.connect(database=lexicon_bdd) as lexicon:
        lexicon_cursor = lexicon.cursor()

    lexicon_cursor.execute("SELECT distinct cgramortho,ortho FROM Lexique order by length(ortho) asc")

    corpus = list(takewhile(cursor=lexicon_cursor))

    # print(corpus)
    # for j, regex in enumerate(procedure_corpus2(corpus=corpus, debug=True)):
    #     print(j, regex)
    #     for i, sequence in enumerate(corpus):
    #         if re.search(regex, sequence):
    #             dico[regex].add(sequence)
    # print(*["\t".join((x, ",".join(y))) for (x,y) in dico.items() if len(y)>1], sep='\n', file=open('temporaire1.txt', 'w', 'utf-8'))
    bdd = psycopg2.connect(dbname='template1')
    # bdd.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    bdd.autocommit = True
    bdd_cursor = bdd.cursor()
    # bdd_cursor.execute("PRAGMA main.synchronous=OFF")
    # bdd_cursor.execute("PRAGMA main.journal_mode=TRUNCATE")
    # bdd_cursor.execute("PRAGMA main.locking_mode=EXCLUSIVE")
    try:
        bdd_cursor.execute(
            "CREATE DATABASE Gestion TABLESPACE=Perceptron ENCODING=UTF-8 TEMPLATE=template0 CONNECTION LIMIT=1"
        )
    except psycopg2.ProgrammingError:
        pass
    bdd_cursor.execute("drop table if exists genealogie")
    bdd_cursor.execute("drop table if exists examples")
    bdd_cursor.execute("drop table if exists features")
    bdd_cursor.execute("drop table if exists classes")
    bdd.autocommit = False
    create_tables(cursor=bdd_cursor)

    procedure_corpus_parallel(
        corpus=corpus,
        bdd_cursor=bdd_cursor,
        seuil=1000000,
        bdd=bdd,
        debug=True
    )
    bdd.close()


def main4():
    bdd = psycopg2.connect(dbname='Memory')
    bdd_cursor = bdd.cursor()
    bdd_cursor.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
    print(bdd_cursor.fetchall())


def takewhile(func=lambda x: (x[0].split(','), x[1]), cursor: sqlite3.Cursor=None):
    i = 1
    if cursor is None:
        return
    tmp = cursor.fetchone()
    while tmp is not None:
        yield (i, func(tmp))
        tmp = cursor.fetchone()
        i += 1


def get_bdd_size(bdd):
    """

    :param bdd:
    :return:
    """
    pass


if __name__ == '__main__':
    main4()
    # parser = ArgumentParser(prog="", description="")
    # parser.add_argument(
    #     "train_set"
    # )
    #
    # parser.add_argument(
    #     "test_set"
    # )
    #
    # parser.add_argument(
    #     "iterMax",
    #     type=int
    # )
    #
    # parser.add_argument(
    #     "-v",
    #     "--verbose"
    # )
    #
    # args = parser.parse_args("ab bc 5".split(' '))
    #
    # train_set = verify_format('train')
    # test_set = verify_format('test')
    #
    # examples = Chest(path="examples")
    # weights = Chest(path="weights")
    # examples_queue = Queue()
    # sequences_queue = Queue()
    # test_queue = Queue()
    # # instanciation des processus
    # process_perceptron = Process(name="perceptron", target=procedure_perceptron, args=(args.iterMax, train_set, examples, examples_queue, sequences_queue))
    # process_example = Process(name="example", target=procedure_example, args=(examples_queue, test_queue, weights))
    # process_powerset = Process(name="powerset", target=procedure_powerset, args=(sequences_queue, examples_queue, examples, train_set))
    # # process_test = Process(target=procedure_test, args=(test_set,))
    # # démarrage des processus
    # process_perceptron.start()
    # process_example.start()
    # process_powerset.start()
    # # process_test.start()
    # # arrêt des processus
    # process_perceptron.join()
    # process_example.join()
    # process_powerset.join()
    # # process_test.join()

