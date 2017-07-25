# coding: utf-8
from re import compile
import sqlite3
import itertools
import yaml
from codecs import open


def gestion_infover(sequence):
    """
        parser le champs infover de lexique 381
        on etourne un dictionnaire avec les champs mode, temps et personne
    :param sequence:
    :return:
    """
    tmp = ('mode', 'temps', 'personne')
    return list(map(lambda z: dict(zip(tmp, z.split(':'))), set(sequence.split(';'))))


def gestion_bdlex(): pass


def gestion_lexique381(ortho='', phon='', lemme='', cgram='', genre='', nombre='', infover='',  cursor=None):
    cgrams = {
        "NOM",
        "AUX",
        "VER",
        "ADV",
        "PRE",
        "ADJ",
        "ONO",
        "CON",
        "ART:def",
        "ADJ:ind",
        "PRO:ind",
        "PRO:int",
        "PRO:rel",
        "ADJ:num",
        "PRO:dem",
        "ADJ:dem",
        "PRO:per",
        "ART:ind",
        "LIA",
        "PRO:pos",
        "ADJ:pos",
        "ADJ:int",
    }

    langues = {
        "francais": {
            "NOM": {
                'gen': 'inher',
                'num': {
                    'sg': 'morph0',
                    'pl': 'affix'
                }
            },
            "AUX": "",
            "VER": "",
            "ADV": "",
            "PRE": "",
            "ADJ": "",
            "ONO": "",
            "CON": "",
            "ART:def": "",
            "ADJ:ind": "",
            "PRO:ind": "",
            "PRO:int": "",
            "PRO:rel": "",
            "ADJ:num": "",
            "PRO:dem": "",
            "ADJ:dem": "",
            "PRO:per": "",
            "ART:ind": "",
            "LIA": "",
            "PRO:pos": "",
            "ADJ:pos": "",
            "ADJ:int": "",
    }
    }

    inherent = "({})"
    morph0 = "[{}]"
    underscpec = ":{}"
    affix = "-{}"
    clitic = ":{}"
    redup = "~{}"
    infix = "-{}"
    circum = ""
    ablaut = "\\{}"
    portmanteau = ".{}"
    phrase = "_{}"
    polys = "/{}"

    if cgram in cgrams:
        if cgram == "NOM":
            genre = inherent.format(genre)
            if nombre == "sg":
                nombre = morph0.format(nombre.upper())
            elif nombre == 'pl':
                nombre = affix.format(nombre.upper())
        elif cgram in ["VER", 'AUX']:
            infover = decoupe(infover)
            if infover:
                if len(infover) > 1:
                    pass
            # personne = personne.upper()
        elif cgram == "ADV":
            pass
        elif cgram == "PRE":
            pass
        elif cgram == "ADJ":
            pass
        elif cgram == "ONO":
            pass
        elif cgram == "CON":
            pass
        elif cgram == "ART:ind":
            pass
        elif cgram == "ADJ:ind":
            pass
        elif cgram == "ADJ:num":
            pass
        elif cgram == "ADJ:dem":
            pass
        elif cgram == "ADJ:pos":
            pass
        elif cgram == "ADJ:int":
            pass
        elif cgram == "PRO:ind":
            pass
        elif cgram == "PRO:int":
            pass
        elif cgram == "PRO:rel":
            pass
        elif cgram == "PRO:dem":
            pass
        elif cgram == "PRO:per":
            pass
        elif cgram == "PRO:pos":
            pass
        elif cgram == "ART:def":
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="gloseur",
        description="programme permettant à partir d'un mot ou d'une base de donnée, de gloser un mot",
    )

    parser.add_argument(
        'mot',
        type=str
    )

    parser.add_argument(
        '-c',
        '--column',
        default='ortho'
    )

    parser.add_argument(
        'bdd',
        type=sqlite3.connect
    )

    parser.add_argument(
        '-t',
        '--table',
        default='lexique'
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true'
    )

    args = parser.parse_args('ce lexique.db'.split(' '))

    curseur = args.bdd.cursor()

    curseur.execute('select distinct phon,cgram,infover,genre,nombre from Lexique where ortho=?', (args.mot,))

    print(curseur.fetchall())
    curseur.execute('select distinct cgram from lexique')
    # curseur.execute('select distinct cgram from lexique')
    print(*list(itertools.chain(*curseur.fetchall())), sep='\n')


def main2():
    bdd = sqlite3.connect('bdlexique.db')
    cursor = bdd.cursor()
    cursor.execute('select distinct ortho from bdlexique where tam="po" and categorie="P"')
    a = sorted(filter(lambda x: x, map(str, itertools.chain(*cursor.fetchall()))))
    print(a)


    # cursor.execute(
    #     """create table bdlexique (
    #         {0} {12},
    #         {1} {12},
    #         {2} {12},
    #         {3} {12},
    #         {4} {12},
    #         {5} {12},
    #         {6} {12},
    #         {7} {12},
    #         {8} {12},
    #         {9} {12},
    #         {10} {12},
    #         {11} {12}
    #     )""".format(*name_champs, 'STRING'))
    # with open('../../Lexiques/BDLexique/bdlexique_repare.txt', mode='r', encoding='utf-8') as entree:
    #     for ligne in entree:
    #         champs = ligne.strip().split(';')
    #         cursor.execute(
    #             "INSERT OR IGNORE INTO bdlexique VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
    #             champs
    #         )
    #     bdd.commit()

if __name__ == '__main__':
    main2()
