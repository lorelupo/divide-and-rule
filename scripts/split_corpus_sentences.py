#!/usr/bin/env python3

import sys
import argparse
import collections
import torch
import os
import re
import numpy as np

def same_antecedent(corefs):
    a_s, a_e = corefs[0][:2]
    for t in corefs:
        if a_s != t[0] or a_e != t[1]:
            return False
    return True

def embedded_corefs(corefs):
    embed = False
    res = None
    for t in corefs:
        if (t[0] <= t[2] and t[1] > t[3]) or (t[0] < t[2] and t[1] >= t[3]):
            embed = True
        elif (t[2] <= t[0] and t[3] > t[1]) or (t[2] < t[0] and t[3] >= t[1]):
            embed = True
        else:
            res = t
    return embed, res

def is_splittable(curr_idx, src, min_length):
    return not(curr_idx < min_length or curr_idx > len(src)-min_length)

def move_split_point(curr_idx, src, min_length, low_bound, up_bound):
    if low_bound == up_bound:
        return curr_idx

    backup = curr_idx
    if curr_idx < min_length:
        assert curr_idx <= up_bound
        while not is_splittable(curr_idx, src, min_length) and curr_idx <= up_bound:
            curr_idx += 1
        if curr_idx <= up_bound:
            return curr_idx
        else:
            return backup
    elif curr_idx > len(src)-min_length:
        assert curr_idx > low_bound
        while not is_splittable(curr_idx, src, min_length) and curr_idx > low_bound:
            curr_idx -= 1
        if curr_idx > low_bound:
            return curr_idx
        else:
            return backup

    return backup

def project_idx(src_idx, src, tgt):
    return int( float(src_idx) / float(len(src)) * len(tgt) )

def generate_split_index(src, tgt, ant_e, ment_s, min_length):
    src_split_idx = (ant_e + ment_s) // 2
    if src_split_idx == ant_e and (ment_s-ant_e) == 1:
        src_split_idx += 1
    tgt_split_idx = project_idx(src_split_idx, src, tgt)
    split_code = 2
    if not is_splittable(src_split_idx, src, min_length):
        new_split_point = move_split_point(src_split_idx, src, min_length, ant_e, ment_s)
        if is_splittable(new_split_point, src, min_length):
            src_split_idx = new_split_point
            tgt_split_idx = project_idx(src_split_idx, src, tgt)
            split_code = 2
        else:
            src_split_idx = len(src) // 2
            tgt_split_idx = len(tgt) // 2
            split_code = 1
    return src_split_idx, tgt_split_idx, split_code

def middle_split_coref(middle_split, low_bound, up_bound):
    return middle_split <= low_bound or middle_split >= up_bound

def apply_split_policy(idx, src, tgt, coref_dict, args): 


    split_code = 0
    min_segment_tokens = args.min_length[0]//2
    if coref_dict is not None and idx in coref_dict:

        if args.verbose:
            print(' ###################################################')
            print(' *** coref_dict[{}]: {}'.format(idx, coref_dict[idx]))
            print(' ###################################################')
            print(' -----')
            print(' Source sentence (length: {}): {}'.format(len(src), src))
            print(' -----')
            for t in coref_dict[idx]:
                print('    * antecedent ({}, {}): {}; mention ({}, {}): {} *'.format(t[0], t[1], src[t[0]:t[1]], t[2], t[3], src[t[2]:t[3]]))
            print(' =====')
            sys.stdout.flush()

        embed_flag, not_embed_coref = embedded_corefs( coref_dict[idx] )
        if embed_flag:
            if args.verbose:
                print(' *** EMBEDDED COREFERENCES !')
                sys.stdout.flush()
            if not_embed_coref is not None:
                if args.verbose:
                    print('     * Found not embedded coreference: {}'.format(not_embed_coref))
                    sys.stdout.flush()
                src_split_idx = (not_embed_coref[1]+not_embed_coref[2])//2
                if src_split_idx == not_embed_coref[1] and (not_embed_coref[2] - not_embed_coref[1]) == 1:
                    src_split_idx += 1
                src_split_idx = move_split_point(src_split_idx, src, min_segment_tokens, not_embed_coref[1], not_embed_coref[2])
                tgt_split_idx = project_idx(src_split_idx, src, tgt)

                if is_splittable(src_split_idx, src, min_segment_tokens):
                    split_code = 3 if middle_split_coref(len(src)//2, not_embed_coref[0], not_embed_coref[3]) else 2
                    return src_split_idx, tgt_split_idx, split_code
                else:
                    return len(src)//2, len(tgt)//2, 1
            else:
                return len(src)//2, len(tgt)//2, 1

        # Apply coreference-based split
        #raise ValueError('Not implemented yet')
        if len(coref_dict[idx]) > 1:    # More than one coreference in the same sentence

            if args.verbose:
                print(' ==================================================')
                print(' *** split-coref policy: multiple coreference event')
                print(' ==================================================')
                sys.stdout.flush()

            if same_antecedent(coref_dict[idx]):    # All the coreferences have the same antecedent

                if args.verbose:
                    print(' *********************')
                    print('   *** same antecedent')
                    print(' *********************')
                    sys.stdout.flush()


                # Try to split the antecedent from the mentions comparing the end index of the antecedent and the start index of the closest mention
                a_e = coref_dict[idx][0][1]
                m_s = min([coref_dict[idx][i][2] for i in range(len(coref_dict[idx]))])

                if args.verbose:
                    print('    * Trying to split between {} - {}'.format(a_e, m_s))
                    sys.stdout.flush()

                assert a_e <= m_s
                #if a_e >= m_s:
                #    return len(src)//2, len(tgt)//2, 1

                src_split_idx, tgt_split_idx, split_code = generate_split_index(src, tgt, a_e, m_s, min_segment_tokens)
                if split_code != 1:
                    split_code = 3 if middle_split_coref(len(src)//2, coref_dict[idx][0][0], m_s+1) else 2

                if args.verbose and (split_code == 2 or split_code == 3):
                    print('     * Succeded, splitting at {}/{}'.format(src_split_idx, tgt_split_idx))
                    sys.stdout.flush()

                if split_code == 1:
                    # Try to split anyway coreferences...

                    if args.verbose:
                        print('    * Failed, trying to split coreferences elsewhere')
                        sys.stdout.flush()

                    last_idx = -1
                    split_t_idx = -1
                    for t_idx, t in enumerate(coref_dict[idx]):
                        src_split_idx = (t[1] + t[2]) // 2
                        if src_split_idx == t[1] and (t[2] - t[1]) == 1:
                            src_split_idx += 1
                        src_split_idx = move_split_point( src_split_idx, src, min_segment_tokens, t[1], t[2] )
                        if is_splittable(src_split_idx, src, min_segment_tokens):
                            last_idx = src_split_idx
                            split_t_idx = t_idx
                    if last_idx > 0:
                        src_split_idx = last_idx
                        tgt_split_idx = project_idx(src_split_idx, src, tgt)
                        split_code = 3 if middle_split_coref(len(src)//2, coref_dict[idx][t_idx][0], coref_dict[idx][t_idx][3]) else 2

                        if args.verbose:
                            print('     * Succeded, splitting at {}/{}'.format(src_split_idx, tgt_split_idx))
                            sys.stdout.flush()
                    else:
                        src_split_idx = len(src) // 2
                        tgt_split_idx = len(tgt) // 2
                        split_code = 1

                        if args.verbose:
                            print('     * Failed, splitting in the middle: {}/{}'.format(src_split_idx, tgt_split_idx))
                            sys.stdout.flush()

                #sys.exit(0)
            else:
                # Try to maximize the number of split coreferences staying close to the middle...

                if args.verbose:
                    print(' ********************************')
                    print('   *** different antecedents case')
                    print(' ********************************')
                    sys.stdout.flush()

                best_margin = -1
                best_idx = 0
                last_idx = -1
                n_split = 0
                best_n_split = 0
                split_t_idx = 0
                best_split_t_idx = 0
                middle_split = len(src) // 2
                for t_idx, t in enumerate(coref_dict[idx]):
                    src_split_idx = (t[1] + t[2]) // 2
                    if src_split_idx == t[1] and (t[2] - t[1]) == 1:
                        src_split_idx += 1
                    src_split_idx = move_split_point( src_split_idx, src, min_segment_tokens, t[1], t[2] )
                    if abs(src_split_idx-middle_split) > best_margin and is_splittable(src_split_idx, src, min_segment_tokens):
                        best_margin = abs(src_split_idx - middle_split)
                        best_n_split += 1
                        best_split_t_idx = t_idx
                        best_idx = src_split_idx
                    elif is_splittable(src_split_idx, src, min_segment_tokens):
                        last_idx = src_split_idx
                        n_split += 1
                        split_t_idx = t_idx
                if best_n_split > 0 and best_n_split >= n_split:
                    src_split_idx = best_idx
                    tgt_split_idx = project_idx(src_split_idx, src, tgt)
                    split_code = 3 if middle_split_coref(len(src)//2, coref_dict[idx][best_split_t_idx][0], coref_dict[idx][best_split_t_idx][3]) else 2

                    if args.verbose:
                        print('     * split margin maximized, splitting at {}/{}'.format(src_split_idx, tgt_split_idx))
                        sys.stdout.flush()
                elif n_split > 0:
                    src_split_idx = last_idx
                    tgt_split_idx = project_idx(src_split_idx, src, tgt)
                    split_code = 3 if middle_split_coref(len(src)//2, coref_dict[idx][split_t_idx][0], coref_dict[idx][split_t_idx][3]) else 2

                    if args.verbose:
                        print('     * coreference split number maximized, splitting at {}/{}'.format(src_split_idx, tgt_split_idx))
                        sys.stdout.flush()
                else:
                    src_split_idx = middle_split
                    tgt_split_idx = len(tgt) // 2
                    split_code = 1

                    if args.verbose:
                        print('     * Maximum margin split Failed, splitting at {}/{}'.format(src_split_idx, tgt_split_idx))
                        sys.stdout.flush()
                #sys.exit(0)

        else:
            if args.verbose:
                print(' =========================================')
                print(' *** split-coref policy: single event case')
                print(' =========================================')

            ant_e = coref_dict[idx][0][1]
            ment_s = coref_dict[idx][0][2]
            #assert ant_e <= ment_s
            if ant_e <= ment_s:
                src_split_idx, tgt_split_idx, split_code = generate_split_index(src, tgt, ant_e, ment_s, min_segment_tokens)
                if split_code != 1:
                    split_code = 3 if middle_split_coref(len(src)//2, coref_dict[idx][0][0], coref_dict[idx][0][3]) else 2
            else:
                src_split_idx = len(src)//2
                tgt_split_idx = len(tgt)//2
                split_code = 1

            if args.verbose:
                print('    * split-code {}, src/tgt split index: {}/{}'.format(split_code, src_split_idx, tgt_split_idx))
                sys.stdout.flush()
    else:
        # Apply split in the middle
        src_split_idx = len(src) // 2
        tgt_split_idx = len(tgt) // 2

    if args.verbose:
        print(' * SO FAR SO GOOD!')
        sys.stdout.flush()
        #sys.exit(0)

    return src_split_idx, tgt_split_idx, split_code


def where_to_split(aligns, m, src_length, tgt_length):
    if m == 0:
        # if it was not possible to cut the sentence around the middle
        # and not too close to the end (only one word in second split)
        # cut roughly in two both source and target
        return src_length // 2, tgt_length // 2, 1
    else:
        cannot_split = True
        m_init = m
        while cannot_split:
            max_before = max(aligns[:m+1,0])
            last_good = np.where(~(max_before < aligns[:,0]))[0][-1]
            if last_good - m > 0:
                m = last_good
            else: 
                cannot_split=False
        src_split = max_before
        tgt_split = aligns[last_good,1]
        # check if there are at least two words in the second sentence
        if (src_length-1)-src_split < 2 or (tgt_length-1)-tgt_split < 2:
            # if not, try to see if possible to cut earlier
            return where_to_split(aligns, m_init -1, src_length, tgt_length)
        else:
            return src_split, tgt_split, 0

def main(args):

    coref_dict = None
    if args.coref_info:
        f = open(args.coref_info)
        data = f.readlines()
        f.close()
        coref_data = [l.rstrip().split() for l in data]
        coref_dict = {}
        for t in coref_data:
            k = int(t[0])
            if k not in coref_dict:
                coref_dict[k] = []
            ant_s, ant_e, ment_s, ment_e = (int(t[3]), int(t[4]), int(t[1]), int(t[2])) # Most frequent case
            if ant_s > ment_s:
                ant_s, ant_e, ment_s, ment_e = (int(t[1]), int(t[2]), int(t[3]), int(t[4]))
            coref_dict[k].append( (ant_s, ant_e, ment_s, ment_e) ) # we store by appearing order

        print('')
        print(' - Read {} coreference events in {} different sentences'.format(len(coref_data), len(coref_dict))) 

    with open(args.src, 'r') as src_in, \
            open(args.tgt, 'r') as tgt_in, \
                open(args.src + '.split', 'w+') as src_out, \
                    open(args.tgt + '.split', 'w+') as tgt_out:
        if args.align:
            middle_splits=0
            with open(args.align, 'r') as alin:
                for a, s, t in zip(alin, src_in, tgt_in) :
                    if s.strip():
                        # line is non-empty
                        s_words = s.split()
                        t_words = t.split()
                        len_s = len(s_words)
                        if len_s >= args.min_length[0]:
                            # extract alignments for current sentence
                            aligns = np.array(
                                [list(map(int, c.split('-'))) for c in a.split()]
                            )
                            # find where we can split the sentence
                            # the closest to the middle, but without breaking
                            # alignment crossings. If we encounter a crossing,
                            # we split it after.
                            m = len(aligns)//2
                            len_t = len(t_words)
                            src_idx, tgt_idx, middle = where_to_split(
                                aligns, m, len_s, len_t)
                            middle_splits += middle
                            # split
                            s1 = s_words[:src_idx+1]
                            s2 = s_words[src_idx+1:]
                            t1 = t_words[:tgt_idx+1]
                            t2 = t_words[tgt_idx+1:]  
                            # if middle:
                            #     print(aligns)
                            #     print(s)
                            #     print(t)
                            #     print("--------")                    
                            src_out.write(' '.join(s1) + '\n')
                            src_out.write(' '.join(s2) + '\n')
                            tgt_out.write(' '.join(t1) + '\n')
                            tgt_out.write(' '.join(t2) + '\n')
                        elif not args.remove_shorter:
                            src_out.write(' '.join(s_words) + '\n')
                            tgt_out.write(' '.join(t_words) + '\n')
                    else:
                        assert s == t, "Src and tgt corpus dont have the same documents' boundaries"
                        # just write empty line
                        src_out.write(s)
                        tgt_out.write(t)
            print("Total middle splits: 2x{0}".format(middle_splits))
        elif args.coref_info:
            # Some stats about the split policy application
            na_split = 0
            zero_split = 0
            one_split = 0
            two_split = 0
            three_split = 0
            for idx, tup in enumerate(list(zip(src_in, tgt_in))) :
                s, t = tup
                if s.strip():
                    # line is non-empty
                    s_words = s.split()
                    t_words = t.split()
                    len_s_words = len(s_words)
                    len_t_words = len(t_words)
                    if len_s_words >= args.min_length[0]:
                        # if sentence length is even, the first half of the split
                        # will have one token less than the second half
                        src_split_idx, tgt_split_idx, split_code = apply_split_policy(idx, s_words, t_words, coref_dict, args)
                        if split_code == 0:
                            zero_split += 1
                        elif split_code == 1:
                            one_split += 1
                        elif split_code == 2:
                            two_split += 1
                        elif split_code == 3:
                            three_split += 1
                        src_out.write(' '.join(s_words[:src_split_idx]) + '\n')
                        src_out.write(' '.join(s_words[src_split_idx:]) + '\n')
                        tgt_out.write(' '.join(t_words[:tgt_split_idx]) + '\n')
                        tgt_out.write(' '.join(t_words[tgt_split_idx:]) + '\n')
                    elif not args.remove_shorter:
                        src_out.write(' '.join(s_words) + '\n')
                        tgt_out.write(' '.join(t_words) + '\n')
                        na_split += 1
                else:
                    assert s == t, "Src and tgt corpus dont have the same documents' boundaries"
                    # just write empty line
                    src_out.write(s)
                    tgt_out.write(t)
                    na_split += 1

            if args.coref_info:
                print(' -----')
                print(' - Split strategy statistics:')
                total_split = zero_split + one_split + two_split + three_split + na_split
                print('   * Total splits: {}'.format(total_split))
                print('   * Exclusive middle splits/not concerned: {}/{} ({:.2f}%/{:.2f}%)'.format(zero_split, na_split, zero_split*100.0/total_split, na_split*100.0/total_split))
                print('   * Middle splits by coref split fails: {} ({:.2f}%)'.format(one_split, one_split*100.0/total_split))
                print('   * Coreference splits: {} ({:.2f}%)'.format(two_split+three_split, (two_split+three_split)*100.0/total_split))
                print('     * out of which {} ({:.2f}%) exclusively by coref split'.format(three_split, three_split*100/total_split))
                print(' ----------')
        else:
            for s, t in zip(src_in, tgt_in) :

                if s.strip():
                    # line is non-empty
                    s_words = s.split()
                    t_words = t.split()
                    len_s_words = len(s_words)
                    len_t_words = len(t_words)
                    if args.verbose:
                        print(' ###################################################')
                        print(' *** Split sentence with standard method')
                        print(' ###################################################')
                        print(' -----')
                        print(' Source sentence (length: {}): {}'.format(len_s_words, s))
                        print(' Target sentence (length: {}): {}'.format(len_t_words, t))
                    if len_s_words >= args.min_length[0] and len_t_words > 1:
                        # SPLIT IN HALF
                        # if sentence length is not a multiple of 2,
                        # the second segment will be the one wih one more token
                        s1 = ' '.join(s_words[:len_s_words//2]) + '\n'
                        s2 = ' '.join(s_words[len_s_words//2:]) + '\n'
                        src_out.write(s1)
                        src_out.write(s2)
                        t1 = ' '.join(t_words[:len_t_words//2]) + '\n'
                        t2 = ' '.join(t_words[len_t_words//2:]) + '\n'
                        tgt_out.write(t1)
                        tgt_out.write(t2)
                        if args.verbose:
                            print(' -----')
                            print('Segment 1')
                            print(' \t Source (length: {}):\n{}'.format(len(s1.split()), s1))
                            print(' \t Target (length: {}):\n{}'.format(len(t1.split()), t1))
                            print('Segment 2')
                            print(' \t Source (length: {}):\n{}'.format(len(s2.split()), s2))
                            print(' \t Target (length: {}):\n{}'.format(len(t2.split()), t2))
                        if len(args.min_length)==2 and len_s_words >= args.min_length[1] :
                            # SPLIT IN 3 SEGMENTS
                            # if sentence length is not a multiple of 3,
                            # the third segment will be the one wih more tokens
                            s1 = ' '.join(s_words[:len_s_words//3]) + '\n'
                            s2 = ' '.join(s_words[len_s_words//3:len_s_words//3*2]) + '\n'
                            s3 = ' '.join(s_words[len_s_words//3*2:]) + '\n'
                            src_out.write(s1)
                            src_out.write(s2)
                            src_out.write(s3)
                            t1 = ' '.join(t_words[:len_t_words//3]) + '\n'
                            t2 = ' '.join(t_words[len_t_words//3:len_t_words//3*2]) + '\n'
                            t3 = ' '.join(t_words[len_t_words//3*2:]) + '\n'              
                            tgt_out.write(t1)
                            tgt_out.write(t2)
                            tgt_out.write(t3)
                            if args.verbose:
                                print(' -----')
                                print('Segment 1')
                                print(' \t Source (length: {}):\n{}'.format(len(s1.split()), s1))
                                print(' \t Target (length: {}):\n{}'.format(len(t1.split()), t1))
                                print('Segment 2')
                                print(' \t Source (length: {}):\n{}'.format(len(s2.split()), s2))
                                print(' \t Target (length: {}):\n{}'.format(len(t2.split()), t2))
                                print('Segment 3')
                                print(' \t Source (length: {}):\n{}'.format(len(s3.split()), s3))
                                print(' \t Target (length: {}):\n{}'.format(len(t3.split()), t3))
                    elif not args.remove_shorter:
                        src_out.write(' '.join(s_words) + '\n')
                        tgt_out.write(' '.join(t_words) + '\n')
                else:
                    assert s == t, "Src and tgt corpus dont have the same documents' boundaries"
                    # just write empty line
                    src_out.write(s)
                    tgt_out.write(t)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Convert a document-level dataset into \
        the same dataset but where each sentence which is longer \
        than a minimum length has been split in two or more segments.'
        )
    # fmt: off
    parser.add_argument(
        '--src', required=True,
        help='Path the source-language document-level corpus.'
        )
    parser.add_argument(
        '--tgt', required=True,
        help='Path the target-language document-level corpus. \
            This corpus must be parallel w.r.t. the source one'
            )
    parser.add_argument(
        '--align', default=None,
        help='Path the alignment file between src and tgt \
        documents, produced with FastAlign.' \
            )
    parser.add_argument(
        '--min-length', 
        nargs="+",
        type=int,
        default=[7],
        help='Sentences shorter than min_lenght[0] will not be splitted;\
            Sentences of at least min_lenght[0] tokens will be split in 2; \
            Sentences of at least min_length[1]tokens  will be split in 3.'
        )
    parser.add_argument(
        '--remove-shorter', default=False, action='store_true',
        help='Wether to remove sentences that are shorter than min_length.'
        )
    parser.add_argument(
        '--coref-info', type=str, default=None,
        help='Coreference information in the format: s_idx, s_m2, e_m2, s_m1, e_m1 (sentence index, mention2 start index, mention2 end index, mention1 (antecedent) start index, mention1 end index)'
    )
    parser.add_argument(
        '--verbose', default=False, action='store_true',
        help='Activate the verbose mode.'
    )
    # fmt: on
    args = parser.parse_args()
    main(args)
