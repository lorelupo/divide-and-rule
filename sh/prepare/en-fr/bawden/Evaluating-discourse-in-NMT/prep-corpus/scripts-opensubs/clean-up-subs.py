#!/usr/bin/python3
# coding: utf-8
import argparse
import re
import gzip
import os
import codecs
# from prob_encodings import latin1

anum = "\wàéèïîôöêûüùœçÉÀÇÔÖÎÏÊŒÈÙÜ"
nanum = "[^"+anum+"]" # non-alphanumerical


# def fixup(m):
#     s = m.group(0)
#     return latin1.get(s, s)
    
# Clean up subtitles

def normalise_punct(text):
    text = re.sub("[–—]", "-", text)   
    text = re.sub("[`´‘]", "'", text) 
    text = text.replace("“", '"')
    text = re.sub("--+", '...', text)
    text = re.sub(r"\\([\'\,])", r"\1", text)
    return text


def detokenise(text, lang):
    if lang=="en": text = re.sub(" \'(s|d|ve|ll|re|m)", r"\'\1", text)
    if lang=="fr": text = re.sub("(l|j|t|s|n|m|c|qu|d)\' ", r"\1\'", text)
    return text

def remove_unwanted_comments(text):
    text = re.sub(r"\(.*?\)", " ", text) # remove anything in brackets
    text = re.sub(r"[\[].*?[\]]", " ", text) # remove anything in brackets
    text = re.sub(r"^[^\[]*?[\]]", " ", text) # remove anything in brackets
    # text = re.sub(r"( -)|(- )", " ", text) # rm some punctuation (when space to one side i.e. bullet-like)
    text = re.sub(r"^-([^\-])", r"\1", text)
    text = re.sub("([^\- ])-$", r"\1", text)
    text = re.sub("[A-Z]+\:", "", text) # sometimes speakers written, eg. PENNY:hello
    # text = re.sub(r"^([A-Z]+ [A-Z]+)+", "", text) # if two words are
    # in caps at beginning, likely to be an error

    for word in ["Man #"]:
        if word in text:
            text=""
            break

    return text

def common_errors(text, lang):
    if lang=="en":
        text = text.replace('l"m', "I'm")
        text = re.sub('([^ ])l" ?II', r"\1I'll", text)
        text = re.sub("(^| )l([',.-])", r"\1I\2", text)
        text = text.replace(" l...l", " I...I")
        text = re.sub("l\-[lI](['\- ])", r"I-I\1", text)
        text = re.sub("'lI[\- ]", "'ll ", text)
        text = re.sub("^lsn't", "Isn't", text)
        text = re.sub(r"^l ", "I ", text)
        text = re.sub(r"^l[tsnf] ", "Is ", text)
        text = re.sub(r"^lt'", "It'", text)
        text = text.replace(" l ", " I ")
        text = re.sub("[\- ]I'lI[\- ]", " I'll ", text)

        for word, replacement in [("belIeve", "believe"), \
                                  ("feelIng","feeling"), \
                                  ("welI", "well"),\
                                  ("wheelI've", "wheel I've"),\
                                  ("LeguelIec", "Leguellec"),\
                                  ("CampbelI", "Campbell"),\
                                  ("telIJoe", "tell Joe"), \
                                  ("tllI", "till"),\
                                  ("y'alI", "y'all"),
                                  ("ﬀ", "ff")]:
            text = text.replace(word, replacement)

    if lang=="fr":
        text = text.replace(r"\xc3'", "ô")
        text = text.replace(r"‡", "ç")
        text = re.sub("\|([ea\'])", r"l\1", text)
        
        # Replace certain words
        if "_" in text:
            for word,rempl in [("di_icile", "difficile"),\
                                ("di_érent","différent"),\
                                ("est_ce", "est-ce"),\
                                ("sou_rir", "sourir"),\
                                ("peut_être", "peut-être"),\
                                ("rendez_vous", "rendez-vous"),
                                ("Avez_vous", "Avez-vous")]:
                text = text.replace(word, rempl)
    # problem of I transcribed as l
    text = re.sub(r"^l([NnLmDdRrTtSsKkFf])", r"i\1", text)
    text = text.replace("¡","i")

    return text

def replace_nonsense_sents(text):
    # just punct
    if re.match("^"+nanum+"+$", text):
        return ""

    # just punct and isolated characters (not always correct, but MOSTLY correct
    if re.match("^"+nanum+"+[b-hj-zB-HJ-Z]"+nanum+"*$", text) or \
        re.match("^"+nanum+"*[b-hj-zB-HJ-Z]"+nanum+"+$", text):
        return ""

    # loads of punct
    if re.match("[\, ;\:\-\'\"\^]{3,}", text):
        return ""

    # If one of these words, ignore sentence
    for symbol in ["==", "__", "elder_man", "fontcolor", "\c", \
                    "'LIAISON'", "''''", "= =", "©"]:
        if symbol in text:
                return ""

    if re.match("^([nlmyi_]|[^"+nanum+"])+$", text):
        return ""
    
    return text

def strange_symbols_final_step(text):
    # Replace problem characters with space
    for symbol in ["_", "^", "\r", "*", "#", "@@", "{\pos }", "\pos",
                   "[" "♪", "[", "]", "♫"]:
        text = text.replace(symbol, " ")

    # Encoding problems
    # if re.search(u"[\x80-\x9f]", text):
    #     os.sys.stderr.write(text)
    #     text = re.sub(u"[\x80-\x9f]", fixup, text)
    #     os.sys.stderr.write(re.sub(u"[\x80-\x9f]", fixup, text))

    
    # Correct latin-1 coding
    # if re.search(b"[\xA1-\xFF\x9c]", bytes(text, "utf-8")):# and not(re.match("^["+anum+"$ \.\,\-\:\;\?\'\"\!\%£$°]+$", text)):
    #     text = re.sub(b"[\xA1-\xFF\x9c]", fixup, bytes(text,"utf-8")).decode("utf-8", "ignore")
    #     os.sys.stderr.write("blah: "+text+"\n")

    # if re.search(b"[\xA1-\xFF\x9c]", bytes(text, "utf-8")):
        # for symbol in latin1:
            # print(bytes(text, "latin-1"))
            # text=bytes(text, "utf-8").replace(symbol,latin1[symbol]).decode("utf-8", "ignore")
        

    for char in [("a","ä","â"),("e","ë","ê"),("i","ï","î"),("o","ö","ô"),\
                ("u","ü","û")]:
        text=re.sub("¨"+char[0], char[1], text)
        text=re.sub("^"+char[0], char[2], text)
        
    # Delete odd symbols
    for symbol in ["^@+", "@$", r"^\++", r"\{", r"\{\\", r"\}", "¤",\
                    r"\\", "¨", "\^", "^ *\- *", "^ *\[", "\] *$"]:
        text = re.sub(symbol, "", text)

    # Pipes
    text = re.sub("\|", " ", text)

    # Only at very end
    if "@" in text: return ""
    
    return text

def clean_subtitle(text, lang):
    text = normalise_punct(text.strip())
    text = detokenise(text, lang)
    text = remove_unwanted_comments(text)
    text = common_errors(text, lang)
    text = replace_nonsense_sents(text)
    text = strange_symbols_final_step(text)

    text = re.sub("  ", " ", text)

    # Tagging bugs
    text = text.replace(" DEV ", " dev ")
    
    return text.strip()

def clean_corpus(infile, lang):
    if ".gz" in infile: fp = gzip.open(infile, "rt", encoding="utf-8")
    else: fp = open(infile, "r", encoding="utf-8")

    for line in fp:
        os.sys.stdout.write(clean_subtitle(line, lang)+"\n")
    fp.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("lang")
    args = parser.parse_args()

    #os.sys.stdout = codecs.getwriter('utf-8')(os.sys.stdout)
    
    clean_corpus(args.infile, args.lang)


