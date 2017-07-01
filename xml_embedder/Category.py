# coding: utf-8

from bs4 import BeautifulSoup

from xml_embedder import Question


class Category(Question.Question):
    def __init__(self, text):
        super(Category, self).__init__()
        self.type = "category"
        self.text = text

    def __repr__(self):
        return self.question.format(type=self.type, body="<cateogry><text>{}</text></category>".format(self.type))

    def __str__(self): return repr(self)


def main():
    x = Category("Glosage")
    with open("text.xml") as tmp:
        f = BeautifulSoup(tmp, 'lxml')
        for q in f.find_all('question'):
            print(q.prettify())

if __name__ == '__main__':
    main()
