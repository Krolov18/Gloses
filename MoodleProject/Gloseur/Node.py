# coding: utf-8


def inverserChiffre(int):
    return 1 if not int else 0


class Node:
    """
        Cette classe est une classe abstraite qui sera héritée
        par des classes plus spécifiques. Elle dit simplement qu'un
        Node doit comporter une batterie d'attributs

        to_string permet de retourner un type string dans un format spécifique à QTree, package de LaTeX
        afin de construire l'arbre de dérivation de la phrase
    """
    def __init__(self, type=None, head=None, children=None, value=None, leaf=None, feedback=None):
        self.type = type
        if children:
            self.children = children
        else:
            self.children = []
        self.head = head
        self.leaf = leaf
        self.value = value
        self.feedback = feedback
        self.to_string = self.to_strings()

    def to_strings(self):
        headstruc = "[ .{0} {1} ]"
        if self.leaf is not None:
            return headstruc.format(self.head,headstruc.format(self.leaf, ""))
        else:
            return headstruc.format(self.head, " ".join([child.to_string for child in self.children]))


class BTOS():
    def __init__(self, bout=None, trait_ordre=None, syntaxe=None, expression=None, comment=None):
        self.B = bout
        self.TO = trait_ordre
        self.S = syntaxe
        self.expression = expression
        self.comment = comment

    def __str__(self):
        print('bouts : ' + ", ".join(self.B))
        print('trait_ordre : ' + ", ".join(self.TO))
        print('syntaxe : ' + ", ".join(self.S))
