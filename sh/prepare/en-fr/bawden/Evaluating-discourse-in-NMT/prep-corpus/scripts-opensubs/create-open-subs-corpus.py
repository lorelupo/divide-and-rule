#!/usr/bin/env python
import os, gzip, re


years=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]

def get_sentence_alignments(sentence_alignments, xml_folder,
                            outputfile, src, tgt):
    # prepare outputs
    en_out = open(outputfile+"."+src, "w")
    fr_out = open(outputfile+"."+tgt, "w")
    if "/" in outputfile:
        directory = outputfile.rsplit("/", 1)[0]
        # print("dir = "+directory)
    else:
        directory = "."
    # filmnum_out = open(directory+"/filminfo.original.list.gz", "wb")
    en_out.write("")
    fr_out.write("")
    # filmnum_out.write("")
    filmnum_out = None

    # to stock docnames and alignments
    imdb = []
    alignments, num_abandoned = [], 0
    current_fromto = ()
    num=0
    filmnum, sentnum = 0, 0

    # read alignment file
    with gzip.open(sentence_alignments, "rb") as align:
        for line in align:
            # detect if beginning of doc and detect id of doc 'from' and 'to'
            line = line.decode("utf-8")
            fromdoc = re.match(r'.*?fromDoc *= *"(.+?)".*?', line)
            todoc = re.match(r'.*?toDoc *= *"(.+?)".*?', line)

            # once source and target docs known
            if fromdoc and todoc:
                # write last doc to file if not the first doc
                if len(current_fromto)==2:
                    imdb_match = re.match(r'.*?'+src+'/\d+/(\d+)/\d+\.xml.gz', current_fromto[0])
                    imdb_match2 = re.match(r'.*?'+tgt+'/(\d+)/(\d+)/\d+\.xml.gz',current_fromto[1])
                    year = imdb_match2.group(1)

                    if imdb_match.group(1) != imdb_match2.group(2):
                        print("Stange... imdb nums are not the same for the two languages")
                        input()

                    # write sentences to files
                    #     - only do an episode once! - take the first one
                    #     - and filter years according to the list above
                    if imdb_match.group(1) not in imdb:# and int(year) in years:
                        filmnum, sentnum = write_sentences(xml_folder, alignments,
                                                           current_fromto[0], current_fromto[1],
                                                           en_out, fr_out, filmnum_out, filmnum, sentnum,
                                                           year, imdb_match2.group(2))
                        imdb.append(imdb_match.group(1))
                    else:
                        if year in years:
                            print("Skipping episode. Episode already included elsewhere.")

                current_fromto= (fromdoc.group(1), todoc.group(1))
                alignments = []
                num+=1

                # keep track of number of documents perused
                if num%100==0:
                    os.sys.stderr.write("\r"+str(num))
                    os.sys.stderr.flush()

            # try to detect new sentence alignments => can't remember what this does!! (TODO)
            target_match = re.match(r'.*?xtargets *="([\d ;]+)".*?', line)
            if target_match and len(current_fromto)==2:
                targets = target_match.group(1)
                en = targets.split(";")[0].split()
                fr = targets.split(";")[1].split()
                if len(en)>0 and len(fr)>0: alignments.append((en, fr))
                else: num_abandoned+=1

            elif target_match:
                print("Uh oh. There was a problem finding the relevant documents")
                input()
            

    # write remaining sentences
    # if not the first time, now write last doc to file
    if len(current_fromto)==2:
        imdb_match = re.match(r'.*?'+src+'/\d+/(\d+)/\d+\.xml.gz', current_fromto[0])
        imdb_match2 = re.match(r'.*?'+tgt+'/\d+/(\d+)/\d+\.xml.gz', current_fromto[1])

        if not imdb_match:
            os.sys.stderr.write("failed imdb match:"+current_fromto[0]+"\n")
        elif not imdb_match:
            os.sys.stderr.write("failed imdb match:"+current_fromto[1]+"\n")
            
        if imdb_match.group(1) != imdb_match2.group(1):
            print("Stange... imdb nums are not the same")
            input()

        # only do an episode once! - and filter by years
        if imdb_match.group(1) not in imdb:# and int(year) in years:
            write_sentences(xml_folder, alignments, current_fromto[0], current_fromto[1],
                            en_out, fr_out, filmnum_out, filmnum, sentnum, year, imdb_match.group(1))
            imdb.append(imdb_match.group(1))
        else:
            if year in years:
                print("Skipping episode. Episode already included elsewhere.")
                
    en_out.close()
    fr_out.close()
    # filmnum_out.close()
    os.sys.stderr.write("Got sentence alignments")
    os.sys.stderr.write("Abandoned "+str(num_abandoned)+" subtitles due to non-alignment")
    return 0


# ---------------------------------------------------------------------------
def read_xml_file(file_pointer):
    sent, sentences = None, []
    sid, i = None,0
    for line in file_pointer:
        line = line.decode()
        sentence_match = re.match(r'.*?<s.*? id="([\d]+)".*?>.*?', line)
        if sentence_match:
            sid = int(sentence_match.group(1))
            i+=1
            if sent!=None:
                sentences.append(sent.strip())
                sent=""
            if sid!=len(sentences)+1:
                print("sentence id = "+str(sid))
                print("number of sentences = "+str(len(sentences)))
                print("There has been an awful error in the sentence id numbers!!!")
                input()
            
        word_match = re.match(r'.*?<w .*?id="[\d\.]+".*?>(.*?)</w>.*?', line)
        if word_match:
            if sent==None: sent=""
            sent += word_match.group(1)+" "
    sentences.append(sent)
    return sentences


# ---------------------------------------------------------------------------
def read_raw_file(file_pointer):

    sent, sentences, meta, startedmeta, startedsubmeta = None, [], {}, False, False
    sid, i = None,0
    for line in file_pointer:
        line = line.decode(encoding="utf-8")
        sentence_match = re.match(r'.*?<s.*? id="([\d]+)".*?>.*?', line)
        if sentence_match:
            sid = int(sentence_match.group(1))
            i+=1
            if sent!=None:
                sent=""
            if sid!=len(sentences)+1:
                print("sentence id = "+str(sid))
                print("number of sentences = "+str(len(sentences)))
                print("There has been an awful error in the sentence id numbers!!!")
                input()
            
        tag_match = re.match(r'.*?<.*?>.*?', line)
        # subtitle text appears on new line (but can be empty)
        # so also add a blank line in this case.
        if not tag_match:
            sent = line.strip()
            sentences.append(sent)
        elif line.strip=="</s>" and sid!=len(sentences):
            sentences.append("")

        # handle meta information
        if "<meta>" in line: startedmeta=True
        elif "</meta>" in line: startedmeta=False
        elif "<subtitle>" in line: startedsubmeta=True
        elif "</subtitle>" in line: startedsubmeta=False
            
        if startedmeta:
            metamatch = re.match(r'[^<>]*<([^<>]+)>([^<>]+)</\1>', line)
            if metamatch:
                if startedsubmeta and metamatch.group(1)=='duration':
                    content = metamatch.group(2).replace("\t", ", ")
                    meta['subduration'] = content
                else:
                    content = metamatch.group(2).replace("\t", ", ")
                    meta[metamatch.group(1)] = content
    # print(meta)
    # raw_input()
    return sentences, meta

# ---------------------------------------------------------------------------
'''
Write sentences to the files pointed by the file pointers en_out and fr_out
'''
def write_sentences(xml_folder, alignments, from_file, to_file,
                    en_out, fr_out, filmnum_out, filmnum, sentnum, year, imdbnum):

    if xml_folder[-1]=="/": xml_folder=xml_folder[:-1]
    try:
        en = gzip.open(xml_folder+"/"+from_file, "rb")
    except IOError:
        os.sys.stderr.write("\nThe requested file does not exist:"+from_file+"\n")
        return
    try:
        fr = gzip.open(xml_folder+"/"+to_file, "rb")
    except IOError:
        os.sys.stderr.write("\nThe requested file does not exist: "+to_file+"\n")
        return
    
    # read both files and stock sentences
    en_sentences, meta = read_raw_file(en)
    fr_sentences, meta_fr = read_raw_file(fr)
    en.close()
    fr.close()

    # print("src num = "+str(len(en_sentences)))
    # print("tgt num = "+str(len(fr_sentences)))
    # print(from_file)
    
    # Get rid of sentences where one is empty
    numsents=0
    for en_sentnums, fr_sentnums in alignments:
        prob_fic = False
        for en_sentnum, fr_sentnum in zip(en_sentnums, fr_sentnums):

            if prob_fic: break
            # print("src sentnum = "+str(en_sentnum))
            # print("tgt sentum = "+str(fr_sentnum))
            if int(en_sentnum) <= len(en_sentences) and int(fr_sentnum) <= len(fr_sentences): #TODO: messy! do not leave this here
                en_text = en_sentences[int(en_sentnum)-1].encode("utf-8").strip()
                fr_text = fr_sentences[int(fr_sentnum)-1].encode("utf-8").strip()

                if en_text=="" or fr_text=="": continue
                else:
                    en_out.write(en_text)
                    fr_out.write(fr_text)
                    # os.sys.stderr.write(en_text)
            else:
                os.sys.stderr.write("\nProblem with sentence numbers!!")
                if int(en_sentnum) > len(en_sentences):
                    os.sys.stderr.write(from_file+"\n")
                if int(fr_sentnum) > len(fr_sentences):
                    os.sys.stderr.write(to_file+"\n")
                    
                # print("src sentnum = "+str(en_sentnum))
                # print("tgt sentum = "+str(fr_sentnum))
                
                prob_fic = True
                continue

                    
        en_out.write("\n".encode("utf-8"))
        fr_out.write("\n".encode("utf-8"))
        
        numsents+=1

    encoding, original, confidence, genre, country, subduration, unk = "","","","","","",""
    if 'encoding' in meta: encoding=meta['encoding']
    if 'original' in meta: original=meta['original']
    if 'confidence' in meta: confidence=meta['confidence']
    if 'genre' in meta: genre=meta['genre']
    if 'country' in meta: country=meta['country']
    if 'subduration' in meta: subduration=meta['subduration']
    if 'unknown_words' in meta: unk=meta['unknown_words']
        
    os.sys.stdout.write("\t".join([str(filmnum+1), str(sentnum+1), str(sentnum+numsents),
                                 imdbnum, year, encoding, original, genre, country,
                                 subduration, unk, confidence, from_file, to_file])+"\n")
    os.sys.stdout.flush()

    return filmnum+1, sentnum+numsents
                

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a parallel corpus from all folders')
    parser.add_argument("-r", "--raw_folder", required=True)
    parser.add_argument("-a", "--alignments", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-s", "--src", required=True)
    parser.add_argument("-t", "--tgt", required=True)
    args = parser.parse_args()

    get_sentence_alignments(args.alignments,
                            args.raw_folder,
                            args.output_file,
                            args.src, args.tgt)
    
