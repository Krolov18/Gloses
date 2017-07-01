# coding: utf-8


class MoodleExo(object):
    def __init__(self):
        super(MoodleExo, self).__init__()
        self.tag = "quiz"
        self.entete = '<?xml version="1.0" ?>'
        self.quiz = "<quiz>{body}</quiz>"

    def __repr__(self):
        return self.entete+"\n"+self._prettify(self.quiz)

    def __str__(self):
        return repr(self)

    def _prettify(self, text):
        import bs4
        return bs4.BeautifulSoup(text, 'lxml').find(self.tag).prettify()

    def add_category(self): pass

    def add_question(self, question, new="{body}"):
        self.quiz = self.quiz.format(body=question + new)


def mots_ligne(line):
    import re
    return re.findall(r"[\wâàéèêëîïôùûüçÂÀÉÈÊËÎÏÔÙÛÜÇæœÆŒ'’=~-]+|[.…,;!?:—–()\[\]/#]", line)


def main():
    import yaml
    from nltk.tokenize import word_tokenize
    x = MoodleExo()
    tmp = """
---
 category: Morphologie/Espace-Thématique
 exo_type: esth
 cloze-markers: rnd,2:sac,sac,sac,sac,sac,sac,sac,sac,sac,sac,sac,sac
 ex:
  - laver: lav,lav,lav,lav,lav,lav,lav,lav,lave,lav6,lava,lave
  - manger: mâZ,mâZ,mâZ,mâZ,mâZ,mâZ,mâZ,mâZ,mâZe,mâZ6,mâZa,mâZe
  - brouter: bRut,bRut,bRut,bRut,bRut,bRut,bRut,bRut,bRute,bRut6,bRuta,bRute
  - garder: gaRd,gaRd,gaRd,gaRd,gaRd,gaRd,gaRd,gaRd,gaRde,gaRd6,gaRda,gaRde
---
 category: Français/Dictées
 exo_type: dictee
 ex:
  - titre: la haine
    son: http://www.ladictee.fr/la_haine.mp3
    text: je parle de haine.
  - titre:
    son:
    text:
  - titre:
    son:
    text:
    """
    r = "name: troupe"
    p = yaml.load_all(tmp)
    print(word_tokenize("papa l'a mangée aujourd'hui."))
    ##for x in p:
    #    print(x)


if __name__ == '__main__':
    main()
