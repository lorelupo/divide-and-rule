#!/usr/bin/python3
import re
import codecs
import os
# coding: utf-8

def filter_sents(src_in, tgt_in, src_out, tgt_out):
    i = 0
    for tgt, src in zip(tgt_in, src_in):
        i+=1
        # empty
        if tgt.strip()=="" or src.strip()=="":
            os.sys.stdout.write(str(i)+"\n")
            continue
        # lines too long
        elif len(src.split()) > 500 or len(tgt.split()) > 500:
            os.sys.stdout.write(str(i)+"\n")
            continue
        else:
            tgt_out.write(tgt.strip()+"\n")
            src_out.write(src.strip()+"\n")


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create a parallel corpus from all folders')
    parser.add_argument("src_in")
    parser.add_argument("tgt_in")
    parser.add_argument("src_out")
    parser.add_argument("tgt_out")
    args = parser.parse_args()

    if ".gz" in args.src_in: src_in = gzip.open(args.src_in, "rt",encoding='utf-8')
    else: src_in = open(args.src_in, "r",encoding='utf-8')

    if ".gz" in args.tgt_in: tgt_in = gzip.open(args.tgt_in, "rt",encoding='utf-8')
    else: tgt_in = open(args.tgt_in, "r",encoding='utf-8')

    if ".gz" in args.src_out: src_out = gzip.open(args.src_out, "rt",encoding='utf-8')    
    else: src_out = open(args.src_out, "w",encoding='utf-8')

    if ".gz" in args.tgt_out: tgt_out = gzip.open(args.tgt_out, "rt",encoding='utf-8')
    else: tgt_out = open(args.tgt_out, "w",encoding='utf-8')
        
    filter_sents(src_in, tgt_in, src_out, tgt_out)
    src_in.close()
    tgt_in.close()
    src_out.close()
    tgt_out.close()    
    
    
