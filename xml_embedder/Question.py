# coding: utf-8


class Question(object):
    __i = -1

    def __init__(self):
        self.tag = "question"
        self.question = '<question type="{type}">{body}</question>'
        self.incremente()

    def __repr__(self):
        tmp = "<!--  question: {0}  -->".format(type(self).__i)
        return tmp+"\n"+self._prettify(self.question)

    def __str__(self):
        return repr(self)

    def incremente(self):
        type(self).__i += 1

    def _prettify(self, text):
        import bs4
        return bs4.BeautifulSoup(text, 'lxml').find(self.tag).prettify()

    def add_type(self, type):
        types = (
            "category",
            "multichoice",
            "cloze",
            "description",
            "essay",
            "matching",
            "truefalse",
            "shortanswer",
            "numerical"
        )
        if type not in types:
            raise TypeError(
                'le type doit être une string faisant partie de {}'.format(
                    ", ".join(types)
                )
            )
        else:
            if type == "category":
                self.question = self.question.format(
                    type=type,
                    body="<category>{0}</category>".format('{body}')
                )
            else:
                self.question = self.question.format(
                    type=type,
                    body="{body}"
                )

    def add_body(self, body):
        self.question = self.question.format(body="<text>{}</text>".format(body))

    def add_name(self, name):
        self.question = self.question.format(
            body="<name><text>{0}</text></name>".format(
                name
            )+"{body}"
        )

    def add_intitule(self, text, formating):
        formats = ('html', 'moodle_auto_format', 'plain_text', 'markdown')
        if formating not in formats:
            raise TypeError('format must be in ({})'.format(", ".join(formats)))
        self.question = self.question.format(
            body="<questiontext format={0}><text>{1}</text></questiontext>".format(
                formating,
                text
            )+"{body}"
        )

    def add_penalty(self, penalty):
        choices = ('1', '0', 1, 0)
        if penalty not in choices:
            raise TypeError('penalty mus be in {}'.format(", ".join(choices)))
        self.question = self.question.format(body="<penalty>{}</penalty>".format(penalty)+"{body}")

    def add_general_feedback(self, fdbck):
        choices = ('1', '0', 1, 0)
        if fdbck not in choices:
            raise TypeError('penalty mus be in {}'.format(", ".join(choices)))
        self.question = self.question.format(body="<generalfeedback>{}</generalfeedback>".format(fdbck)+"{body}")

    def add_defaultgrade(self, deflt):
        choices = ('1', '0', 1, 0)
        if deflt not in choices:
            raise TypeError('penalty mus be in {}'.format(", ".join(choices)))
        self.question = self.question.format(body="<generalfeedback>{}</generalfeedback>".format(deflt) + "{body}")

    def add_hidden(self, hid):
        choices = ('1', '0', 1, 0)
        if hid not in choices:
            raise TypeError('penalty mus be in {}'.format(", ".join(choices)))
        self.question = self.question.format(body="<generalfeedback>{}</generalfeedback>".format(hid) + "{body}")

    def add_reponse(self, reponse, feedback, fraction):
        fdb = "<feedback>{body}<feedback>".format(body=self.add_body(feedback))
        rep = '<answer fraction="{fraction}">{body}</answer>'.format(fraction=fraction, body=self.add_body(fdb))


def xml_struc(tag, body, **options):
    options = options.get('options', options)
    return "<{tag}{options}>{body}</{tag}>".format(
        tag=tag,
        body=body,
        options=" " + " ".join(['{}="{}"'.format(x, y) for x, y in options.items()]) if options else ""
    )


def xml_quiz(body="", **options):
    return xml_struc(tag="quiz", body=body, options=options)


def xml_question(name="", questiontext="", **options):
    return xml_struc(tag="question", body=body, options=options)


def xml_category(body="", **options):
    return xml_struc(tag="category", body=xml_text(body=body), options=options)


def xml_text(body="", **options):
    return xml_struc(tag="text", body=body, options=options)


def xml_name(body="", **options):
    return xml_struc(tag="name", body=xml_text(body=body), options=options)


def xml_questiontext(body="", **options):
    return xml_struc(tag="questiontext", body=xml_text(body=body), options=options)


def xml_single(body="", **options):
    return xml_struc(tag="single", body=body, options=options)


def xml_shuffleanswers(body="", **options):
    return xml_struc(tag="shuffleanswers", body=body, options=options)


def xml_feedback(body="", **options):
    return xml_struc(tag="feedback", body=xml_text(body=body) if body else "<text/>", options=options)


def xml_correctfeedback(body="", **options):
    return xml_struc(tag="correctfeedback", body=xml_text(body=body) if body else "<text/>", options=options)


def xml_partiallycorrectfeedback(body="", **options):
    return xml_struc(tag="partiallycorrectfeedback", body=xml_text(body=body) if body else "<text/>", options=options)


def xml_incorrectfeedback(body="", **options):
    return xml_struc(tag="incorrectfeedback", body=xml_text(body=body) if body else "<text/>", options=options)


def xml_answernumbering(body="", **options):
    if body not in ('none', 'abc', 'ABC', '123'):
        raise TypeError('Attention, answernumbering est un élément de (none, abc, ABC, 123)')
    return xml_struc(tag="answernumbering", body=body, options=options)


def xml_answer(body="", fdbck=("feedback", ""), **options):
    rep = xml_text(body=body)
    fdb = globals().get('xml_{}'.format(fdbck[0]), None)(fdbck[1])
    if fdb is None:
        raise TypeError('fdbck doit être dans (correctfeedback, incorrectfeedback, partiallycorrectfeedback)')
    return xml_struc(tag="answer", body=rep+fdb, options=options)


def xml_GLOSSARY(body="", **options):
    return xml_struc(tag="GLOSSARY", body=body, options=options)


def xml_INFO(body="", **options):
    return xml_struc(tag="INFO", body=body, options=options)


def xml_STUDENTCANPOST(body="", **options):
    return xml_struc(tag="STUDENTCANPOST", body=body, options=options)


def xml_ALLOWDUPLICATEDENTRIES(body="", **options):
    return xml_struc(tag="ALLOWDUPLICATEDENTRIES", body=body, options=options)


def xml_DISPLAYFORMAT(body="", **options):
    return xml_struc(tag="", body=body, options=options)


def xml_SHOWSPECIAL(body="", **options):
    return xml_struc(tag="DISPLAYFORMAT", body=body, options=options)


def xml_SHOWALPHABET(body="", **options):
    return xml_struc(tag="SHOWALPHABET", body=body, options=options)


def xml_SHOWALL(body="", **options):
    return xml_struc(tag="SHOWALL", body=body, options=options)


def xml_ALLOWCOMMENTS(body="", **options):
    return xml_struc(tag="", body=body, options=options)


def xml_USEDYNALINK(body="", **options):
    return xml_struc(tag="ALLOWCOMMENTS", body=body, options=options)


def xml_DEFAULTAPPROVAL(body="", **options):
    return xml_struc(tag="DEFAULTAPPROVAL", body=body, options=options)


def xml_GLOBALGLOSSARY(body="", **options):
    return xml_struc(tag="GLOBALGLOSSARY", body=body, options=options)


def xml_ENTBYPAGE(body="", **options):
    return xml_struc(tag="ENTBYPAGE", body=body, options=options)


def xml_ENTRIES(body="", **options):
    return xml_struc(tag="ENTRIES", body=body, options=options)


def xml_ENTRY(body="", **options):
    return xml_struc(tag="ENTRY", body=body, options=options)


def xml_CONCEPT(body="", **options):
    return xml_struc(tag="CONCEPT", body=body, options=options)


def xml_DEFINITION(body="", **options):
    return xml_struc(tag="DEFINITION", body=body, options=options)


def xml_FORMAT(body="", **options):
    return xml_struc(tag="FORMAT", body=body, options=options)


def xml_CASESENSITIVE(body="", **options):
    return xml_struc(tag="CASESENSITIVE", body=body, options=options)


def xml_FULLMATCH(body="", **options):
    return xml_struc(tag="FULLMATCH", body=body, options=options)


def xml_TEACHERENTRY(body="", **options):
    return xml_struc(tag="TEACHERENTRY", body=body, options=options)


def type_category(body=""):
    return xml_question(body=xml_category(body=body), type="category")


def type_multichoice(answers, single=False, shuffleanswers=1, correctfeedback=None, partiallycorrectfeedback=None, incorrectfeedback=None, feedback=None):
    return xml_question(body=body, type="multichoice")


def type_truefalse(): pass


def type_shortanswer(): pass


def type_cloze(): pass


def type_matching(): pass


def type_essay(): pass


def type_numerical(): pass


def type_description(): pass


def xml_():
    return


def xml_CDATA(body=""):
    return "<![CDATA[{body}]]>"

# def xml_(body="", **options): pass


def main():
    x = Question()
    print(x)


if __name__ == '__main__':
    main()
