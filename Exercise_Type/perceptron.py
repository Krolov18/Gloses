# coding: utf-8
from chest import Chest
from multiprocessing import Process, Pool, Queue, Manager
from collections import defaultdict
from Exercise_Type.Powerset import powerset_tostring
from Exercise_Type.combinations import combinations
from Exercise_Type.Powerset import Variable
from Exercise_Type.tfidf import tf_idf
from sys import stderr
from itertools import chain
import sqlite3
from codecs import open
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


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


def procedure_test(test_set):
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


def maj(sequences, classes, features, data, cursor, debug=False):
    add_value_cmd = "INSERT INTO Examples(example_id,class_id,feature_id,value) VALUES (?,?,?,0.0)"
    select_ids = 'SELECT Classes.id,Features.id from Classes,Features WHERE Classes.classe=? and Features.feature=?'
    add_classes_cmd = "INSERT OR IGNORE INTO Classes(classe) VALUES (?)"
    add_features_cmd = "INSERT OR IGNORE INTO Features(feature) VALUES (?)"
    add_sequence_cmd = "INSERT OR IGNORE INTO Corpus(sequence) VALUES (?)"

    to_tuple = lambda x: (x,)
    to_tuple_seq = lambda t: map(to_tuple, iter(t))
    if debug:
        print("màj temporaire", file=stderr)

    # màj des séquences du corpus
    if debug:
        print("màj corpus", file=stderr)
    cursor.executemany(add_sequence_cmd, to_tuple_seq(sequences))
    sequences.clear()

    # màj des classes
    if debug:
        print("màj classes", file=stderr)
    cursor.executemany(add_classes_cmd, to_tuple_seq(classes))
    classes.clear()

    # màj des features
    if debug:
        print("màj features", file=stderr)
    cursor.executemany(add_features_cmd, to_tuple_seq(features))
    features.clear()

    # màj des examples
    if debug:
        print("màj examples", file=stderr)
    example_numbers = map(lambda x: (x[0],), iter(data))
    ids = map(lambda x: x[1:], iter(data))
    cursor.executemany(
        add_value_cmd,
        list(
            map(
                lambda x: x[0] + x[1],
                zip(
                    example_numbers,
                    map(
                        lambda cc: cursor.execute(select_ids, cc).fetchone(),
                        ids
                    )
                )
            )
        )
    )


def alpha(i: int, j: int, classe: str, regex: str, sequence: str, seuil: int, sequences: set, classes: set,
          features: set, data: set, bdd, cursor, debug: bool=False):
    # if debug:
    #     print("alpha: ", i, j, classe, regex, file=stderr)
    if i == seuil:
        maj(sequences=sequences, classes=classes, features=features, data=data, cursor=cursor, debug=False)
        bdd.commit()
    classes.add(classe)
    features.add(regex)
    sequences.add(sequence)
    sortie = (j, classe, regex)
    data.add(sortie)


def procedure_corpus(corpus, bdd, cursor, seuil: int = 10000,
                     debug: bool = False) -> None:
    i = 0
    data = set()
    classes = set()
    features = set()
    sequences = set()

    for (j, (classe, sequence)) in enumerate(corpus, start=1):
        if debug:
            print("procedure: ", i, j, len(corpus)-j, classe, sequence, file=stderr)
        with Pool(10) as proc:
            args = map(lambda x: (sequence, x, Variable('.'), Variable('.+'), True), combinations(len(sequence)))
            for regex in proc.starmap(powerset_tostring, list(args)):
                alpha(
                    i=i,
                    j=j,
                    classe=classe,
                    regex=regex,
                    sequence=sequence,
                    seuil=seuil,
                    sequences=sequences,
                    classes=classes,
                    features=features,
                    data=data,
                    bdd=bdd,
                    cursor=cursor,
                    debug=True
                )
                if i == seuil:
                    i = 0
                i += 1
    maj(sequences, classes, features, data, cursor, debug=False)


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
    id INTEGER PRIMARY KEY,
    example_id INTEGER,
    class_id INTEGER,
    feature_id INTEGER,
    value REAL,
    foreign key (class_id) REFERENCES Classes(id)
    foreign key (feature_id) REFERENCES Features(id)
)"""
    classes = """CREATE TABLE IF NOT EXISTS Classes (
    id INTEGER PRIMARY KEY,
    classe TEXT UNIQUE
)
"""
    features = """CREATE TABLE IF NOT EXISTS Features (
    id INTEGER PRIMARY KEY,
    feature TEXT UNIQUE
)
"""
    corpus = """CREATE TABLE IF NOT EXISTS Corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence STRING UNIQUE
)
"""
    cursor.execute(examples)
    cursor.execute(classes)
    cursor.execute(features)
    cursor.execute(corpus)


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
    perceptron_bdd = '/Volumes/RESEARCH/Research/Perceptron.db'
    lexicon_bdd = 'Lexique.db'
    lexicon = sqlite3.connect(database=lexicon_bdd)
    lexicon_cursor = lexicon.cursor()
    lexicon_cursor.execute("SELECT * FROM sqlite_master")
    # print(lexicon_cursor.fetchall()[0][-1])

    eee = """CREATE TABLE IF NOT EXISTS Lexique381 (
    ortho TEXT,
    phon TEXT,
    lemme TEXT,
    cgram TEXT,
    genre TEXT,
    nombre TEXT,
    freqlemfilms2 TEXT,
    freqlemlivres TEXT,
    freqfilms2 TEXT,
    freqlivres TEXT,
    infover STRNG,
    nbhomogr TEXT,
    nbhomoph TEXT,
    islem TEXT,
    nblettres TEXT,
    nbphons TEXT,
    p_cvcv TEXT,
    voisorth TEXT,
    voisphon TEXT,
    puorth TEXT,
    puphon TEXT,
    syll TEXT,
    nbsyll TEXT,
    cv_cv TEXT,
    orthrenv TEXT,
    phonrenv TEXT,
    orthosyll TEXT,
    cgramortho TEXT,
    deflem TEXT,
    defobs TEXT,
    old20 TEXT,
    pld20 TEXT,
    morphoder TEXT,
    nbmorph TEXT
)"""
    lexicon_cursor.execute("SELECT cgram,ortho FROM Lexique")
    corpus = lexicon_cursor.fetchall()
    bdd = sqlite3.connect(perceptron_bdd, timeout=10)
    bdd_cursor = bdd.cursor()

    bdd_cursor.execute('pragma main.pagesize=4096')
    bdd_cursor.execute('pragma main.cache_size=-1000000')
    bdd_cursor.execute('pragma main.locking_mode=EXCLUSIVE')
    bdd_cursor.execute('pragma main.synchronous=NORMAL')
    bdd_cursor.execute('pragma main.journal_mode=WAL')
    bdd_cursor.execute('pragma main.cache_size=5000')
    create_tables(cursor=bdd_cursor)
    bdd.commit()
    sequences = list(map(lambda x: x[1], corpus))
    bdd_cursor.execute('SELECT DISTINCT example_id FROM Examples order by example_id asc')
    # print(bdd_cursor.fetchone())
    # tfidf_queue = Queue()
    procedure_corpus(corpus=corpus, bdd=bdd, cursor=bdd_cursor, seuil=100, debug=True)
    # tfidf_process = Process(target=procedure_tfidf, args=(tfidf_queue, bdd, bdd_cursor, sequences, True))
    # tfidf_process.start()
    # tfidf_process.join()


def get_bdd_size(bdd):
    """

    :param bdd:
    :return:
    """
    pass


if __name__ == '__main__':
    main3()
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

