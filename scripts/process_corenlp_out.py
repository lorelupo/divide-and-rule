
import os
import sys
import re

file_lst = sys.argv[1]
f = open(file_lst, encoding='utf-8')
lines = f.readlines()
f.close()

print(' - Read a list of {} files'.format(len(lines)))
sys.stdout.flush()

def process_corenlp_file(filename):

    f = open(filename.rstrip(), encoding='utf-8')
    corenlp_out = f.readlines()
    f.close()

    avg_sent_len = 0.0
    total_sents = 0
    avg_segment_len = 0.0
    total_segments = 0

    l_idx = 0
    sent_flag = False
    dep_flag = False
    coref_flag = False
    sents = []
    file_data = {}  # will contain 2 entries: 1. sentence level data (sentence length and parse tree) 2. coreference data
    sent_data = {}  # will contain sentence level data: 1. sentence length, 2. parse tree
    while l_idx < len(corenlp_out):

        if 'Sentence #' in corenlp_out[l_idx].rstrip():
            l_idx += 1
            sent = corenlp_out[l_idx].rstrip()
            sent_flag = True 

            if 'len' in sent_data:
                sents.append( sent_data )
                sent_data = {}
            l = len(sent.split())
            sent_data['len'] = l
            avg_sent_len += l
            total_sents += 1
            avg_segment_len += l
            total_segments += 2
            l_idx += 1

        if 'Dependency Parse' in corenlp_out[l_idx].rstrip():
            l_idx += 1
            parse = []
            while(len(corenlp_out[l_idx].rstrip()) > 0):
                parse.append( corenlp_out[l_idx].rstrip() )
                l_idx += 1 
            if len(parse) > 0:
                sent_data['parse'] = parse

        if 'Coreference set:' in corenlp_out[l_idx].rstrip():
            l_idx += 1
            coref = []
            while( l_idx < len(corenlp_out) ):
                data_line = corenlp_out[l_idx].rstrip()

                if len(data_line) > 0 and not 'Coreference set:' in data_line: 
                    coref.append( data_line )
                l_idx += 1
            file_data['coref'] = coref

        l_idx += 1

    if 'len' in sent_data:
        sents.append(sent_data)
    file_data['sent'] = sents

    #print(' - Average sentence length: {:.2f} / {} = {:.2f}'.format(avg_sent_len, total_sents, avg_sent_len / total_sents))
    #print(' - Average segment length: {:.2f} / {} = {:.2f}'.format(avg_segment_len, total_segments, avg_segment_len / total_segments))

    return file_data, avg_sent_len, avg_segment_len, total_sents, total_segments

def parse_el_idx(element):

    toks = element.split('-')
    idx_str = toks[-1]

    digit = re.compile('([0-9]+)')
    res = digit.match(idx_str)
    if res == None:
        sys.stderr.write(' - parse_dep_idx: no valid index match in >>>{}<<<'.format(dep))
        sys.exit(1)
    idx = int( res.group(1) )

    return idx

def parse_dep_idx(dep):
    toks = dep.split(',')
    ttoks = toks[-1].split('-')
    idx_str = ttoks[-1].split(')')[0]

    digit = re.compile('([0-9]+)')
    res = digit.match(idx_str)
    if res == None:
        sys.stderr.write(' - parse_dep_idx: no valid index match in >>>{}<<<'.format(dep))
        sys.exit(1)
    idx = int( res.group(1) )

    return idx

def analyse_parse( parse, s_len ):
    '''Example of text output:

        root(ROOT-0, located-5)
        det(University-3, The-1)
        compound(University-3, Stanford-2)
        nsubj:pass(located-5, University-3)
        aux:pass(located-5, is-4)
        case(California-7, in-6)
        obl:in(located-5, California-7)
        punct(located-5, ,-8)
        nsubj(knows-10, Everybody-9)
        csubj(university-15, knows-10)
        obj(knows-10, it-11)
        cop(university-15, is-12)
        det(university-15, a-13)
        amod(university-15, great-14)
        parataxis(located-5, university-15)
        punct(university-15, ,-16)
        acl(university-15, founded-17)
        case(1891-19, in-18)
        obl:in(founded-17, 1891-19)
        punct(located-5, .-20)
    '''

    dep_pattern = re.compile('(\S+)\((\S+)\,\s*(\S+)\)')

    total = 0
    at_least_one = 0
    both = 0
    if s_len <= 6:
        return (0, 0, 1, {})

    root = None
    root_idx = -1
    dependencies_idx = []
    sub_obj_idx = []

    target_deps_idx = {}
    target_deps_idx['subj'] = []
    target_deps_idx['obj'] = []
    for p in parse:
        res = dep_pattern.match(p)
        if res == None:
            sys.stderr.write(' - analyse_parse: no match at dependency >>>{}<<<'.format(p))
            sys.exit(1)
        dname, head, dependent = res.group(1), res.group(2), res.group(3)
        if dname == 'root':
            root = dependent
            root_idx = parse_el_idx(root) 
        elif head == root and dname != 'punct':
            el_idx = parse_el_idx(dependent)
            dependencies_idx.append( (el_idx, dname) )
            if 'nsubj' in dname or 'obj' in dname:
                sub_obj_idx.append( (el_idx, dname) ) 
        if 'root(' in p:
            idx = parse_dep_idx(p) 
            target_deps_idx['root'] = idx 
        if 'nsubj' in p:
            idx = parse_dep_idx(p)
            target_deps_idx['subj'].append( idx ) 
        if 'obj' in p:
            idx = parse_dep_idx(p)
            target_deps_idx['obj'].append( idx ) 

    bound = s_len // 2
    for s in target_deps_idx['subj']:
        assert type(s) == int
        if (s < bound and target_deps_idx['root'] >= bound) or (s >= bound and target_deps_idx['root'] < bound):
            at_least_one = 1
            break

    subj_flag = False
    if at_least_one == 1:
        subj_flag = True
    at_least_one = 0
    for s in target_deps_idx['obj']:
        if (s < bound and target_deps_idx['root'] >= bound) or (s >= bound and target_deps_idx['root'] < bound):
            at_least_one = 1
            break
    if at_least_one == 1 and subj_flag:
        both = 1
    if subj_flag:
        at_least_one = 1
    total = 1   # not used for now, it will correspond to the number of sentences

    at_least_one = 0
    for (idx, name) in sub_obj_idx:
        if (idx < bound and root_idx >= bound) or (idx >= bound and root_idx < bound):
            at_least_one = 1
            break

    both = 0
    split_dep_by_name = {}
    for (idx, name) in dependencies_idx:
        if (idx < bound and root_idx >= bound) or (idx >= bound and root_idx < bound):
            both = 1
            if name not in split_dep_by_name:
                split_dep_by_name[name] = 0
            split_dep_by_name[name] += 1
            break

    return (at_least_one, both, total, split_dep_by_name)

def analyse_coref( coref, s_data, offset ):
    '''Example of text output:

       (1,11,[11,12]) -> (1,3,[1,4]), that is: "it" -> "The Stanford University" 
    '''

    def swap(v1, v2):
        return v2, v1

    def get_split_distance(i, j, m1, m2, sent_data, threshold):

        assert i > j, 'get_split_distance expects different sentence indices, with i > j (got not {} > {})'.format(i,j)

        s1_len = sent_data[i]['len']
        bound1 = s1_len // 2 
        s2_len = sent_data[j]['len']
        bound2 = s2_len // 2
        middle_lens = []
        for s_idx in range(j+1,i):
            middle_lens.append( sent_data[s_idx]['len'] )

        nonsplit_dist = i-j # == 0: ---|----    1: ---|---- ---|----    2: ---|---- ---|---- ---|----
        distance = 0
        ntokens = s1_len + sum( middle_lens ) + s2_len
        if m1 >= bound1:
            if m2 >= bound2:
                distance = 2*(nonsplit_dist)
                ntokens -= bound2
            else:
                distance = 2*(nonsplit_dist) + 1
        else:
            if m2 >= bound2:
                distance = 2*(nonsplit_dist) - 1
                ntokens -= (s1_len-bound1 + bound2)
            else:
                distance = 2*(nonsplit_dist)
                ntokens -= (s1_len-bound1)

        if s1_len <= threshold and m1 >= bound1:
            distance -= 1 
        if s2_len <= threshold and m2 < bound2:
            distance -= 1
        for s_len in middle_lens:
            if s_len <= threshold:
                distance -= 1
        if distance < nonsplit_dist:
            distance = nonsplit_dist

        return distance, ntokens

    def set_coref_stats(d, coref_stats):

        set_idx = 0
        if d <= 3:
            coref_stats[d] += 1
            set_idx = d
        elif d > 3 and d <= 6:
            coref_stats[4] += 1
            set_idx = 4
        elif d > 6 and d <= 10:
            coref_stats[5] += 1
            set_idx = 5
        elif d > 10:
            coref_stats[6] += 1
            set_idx = 6

        return set_idx

    len_threshold = 6
    pronoun_flag = False
    document_size = 4
    pronouns = 'I me my he him she her it we they them his hers its their theirs himself herself itself themselves this that these those who whom which whose'.split()
    #fr_ambig_pronouns = 'you it they my your his her its their mine yours hers theirs him them this that these those whom which whose'.split()
    #pronouns = fr_ambig_pronouns
    coref_pattern = re.compile('\s+\(([0-9]+)\,([0-9]+)\,\[([0-9]+)\,([0-9]+)\]\)\s+\-\>\s+\(([0-9]+)\,([0-9]+)\,\[([0-9]+)\,([0-9]+)\]\)\,\s+that is\:\s+"([\S\s]+)"\s+\-\>\s+"([\S\s]+)"')

    split_coref = 0
    coref_at = [0 for i in range(7)]    # 7 counters for coref at: 0 (same sentence), 1 sentence away, 2, 3, between 3 and 6, between 6 and 10, over 10 
    orig_tokens_at = [0 for i in range(7)]
    split_coref_at = [0 for i in range(7)]
    split_tokens_at = [0 for i in range(7)]
    coref_split_info = []
    for c in coref:
        res = coref_pattern.match(c)
        if res == None:
            sys.stderr.write(' - analyse_coref: incorrect coreference format in >>>{}<<<'.format(c))
            sys.exit(1)

        s1_idx, m1_idx, start1, end1, s2_idx, m2_idx, start2, end2, m1, m2 = int(res.group(1))-1, int(res.group(2))-1, int(res.group(3))-1, int(res.group(4))-1, int(res.group(5))-1, int(res.group(6))-1, int(res.group(7))-1, int(res.group(8))-1 , res.group(9), res.group(10)
        if s1_idx < s2_idx:
            s1_idx, s2_idx = swap(s1_idx, s2_idx)
            m1_idx, m2_idx = swap(m1_idx, m2_idx)
            start1, start2 = swap(start1, start2)
            end1, end2 = swap(end1, end2)

        if not pronoun_flag or (m1.lower() in pronouns or m2.lower() in pronouns):
            if s1_idx == s2_idx:
                coref_at[0] += 1 
                coref_split_info.append( (s1_idx+offset, start1, end1, start2, end2) )
                s_len = s_data[s1_idx]['len']
                orig_tokens_at[0] += s_len

                #print(' - Adding {} tokens at 0'.format(s_len))
                #sys.stdout.flush()

                if s_len > len_threshold:
                    bound = s_len // 2
                    if (m1_idx < bound and m2_idx >= bound) or (m1_idx >= bound and m2_idx < bound):
                        split_coref += 1
                        split_coref_at[1] += 1
                        split_tokens_at[1] += s_len
                    elif (m1_idx >= bound and m2_idx >= bound) or (m1_idx < bound and m2_idx < bound):
                        split_coref_at[0] += 1
                        if m1_idx < bound and m2_idx < bound:
                            split_tokens_at[0] += bound
                        else:
                            split_tokens_at[0] += s_len-bound
                    else:
                        print(' *** I don\'t understand how we can go in this branch!')
                        sys.stdout.flush()
                else:
                    split_coref_at[0] += 1
                    split_tokens_at[0] += s_len
            elif (s1_idx // document_size) == (s2_idx // document_size):
                distance = abs(s1_idx - s2_idx)
                set_idx = set_coref_stats(distance, coref_at)
                for s_idx in range(s2_idx,s1_idx+1):
                    orig_tokens_at[set_idx] += s_data[s_idx]['len']
                split_distance, ntokens = get_split_distance(s1_idx, s2_idx, m1_idx, m2_idx, s_data, len_threshold)
                set_idx = set_coref_stats(split_distance, split_coref_at)
                split_tokens_at[set_idx] += ntokens

    return (split_coref_at, coref_at, split_tokens_at, orig_tokens_at, coref_split_info)

data = {}
total_sent = 0
avg_sent_len, avg_seg_len = 0.0, 0.0
tot_sents, tot_segs = 0, 0
chunkid_re = re.compile('\S+chunk([0-9]+).txt.out')
for filename in lines:
    print('   - processing file {} ...'.format(filename.rstrip()))
    sys.stdout.flush()
    res = chunkid_re.match(filename)
    if res == None:
        sys.stderr.write(' ERROR: cannot parse chunk ID from file name {}'.format(filename) + "\n")
        sys.exit(1)
    chunk_id = int(res.group(1))    
    if chunk_id in data:
        sys.stderr.write(' ERROR: chunk ID {} is already defined'.format(chunk_id) + "\n")
        sys.exit(1)

    file_data, sent_lens, seg_lens, nsents, nsegs  = process_corenlp_file(filename)
    avg_sent_len += sent_lens
    avg_seg_len += seg_lens
    tot_sents += nsents
    tot_segs += nsegs
    #data.append( file_data )

    total_sent += len(file_data['sent'])
    data[chunk_id] = file_data

print(' - Processed {} sentences, parsing and analysing data...'.format(total_sent))
print('   - avg. sentence length: {:.2f} / {} = {:.2f}'.format(avg_sent_len, tot_sents, avg_sent_len / tot_sents))
print('   - avg. segment length: {:.2f} / {} = {:.2f}'.format(avg_seg_len, tot_segs, avg_seg_len / tot_segs))
sys.stdout.flush()

coref_split_info = []
split_corefs_at = [0 for i in range(7)]
corefs_at = [0 for i in range(7)]
split_tokens_at = [0 for i in range(7)]
orig_tokens_at = [0 for i in range(7)]
parse_tot, parse_at_least_one, parse_both = (0, 0, 0)
split_dep_by_name = {}
offset = 0
for k in sorted(data.keys()): 
    d = data[k]
    if 'sent' in d:
        for s in d['sent']:
            if 'parse' in s:
                parse_stats = analyse_parse( s['parse'], s['len'] )
                parse_at_least_one += parse_stats[0]
                parse_both += parse_stats[1]
                parse_tot += parse_stats[2]
                for name in parse_stats[3].keys():
                    if name not in split_dep_by_name:
                        split_dep_by_name[name] = 0
                    split_dep_by_name[name] += parse_stats[3][name]

    if 'coref' in d and 'sent' in d:
        coref_stats = analyse_coref( d['coref'], d['sent'], offset )
        #split_coref += coref_stats[0][1]
        for i in range(7):
            split_corefs_at[i] += coref_stats[0][i]
            corefs_at[i] += coref_stats[1][i]
            split_tokens_at[i] += coref_stats[2][i]
            orig_tokens_at[i] += coref_stats[3][i]
        coref_split_info.extend( coref_stats[4] )
    offset += len(d['sent'])
print(' - done.')
print('')
print(' 1. Found {} sentences out of {} where at least one subj/obj is split from the root'.format(parse_at_least_one, parse_tot))
print(' 2. Found {} sentences out of {} where both, at least once, subj and obj are split from the root'.format(parse_both, parse_tot))
print(' ---')
print(' 3. Found {} ({} in split data) coreference(s) at distance (d) 0'.format(corefs_at[0], split_corefs_at[0]))
print(' 4. Found {} ({} in split data) coreference(s) at distance (d) 1'.format(corefs_at[1], split_corefs_at[1]))
print(' 5. Found {} ({} in split data) coreference(s) at d 2'.format(corefs_at[2], split_corefs_at[2]))
print(' 6. Found {} ({} in split data) coreference(s) at d 3'.format(corefs_at[3], split_corefs_at[3]))
print(' 7. Found {} ({} in split data) coreference(s) at 3 < d <= 6'.format(corefs_at[4], split_corefs_at[4]))
print(' 8. Found {} ({} in split data) coreference(s) at 6 < d <= 10'.format(corefs_at[5], split_corefs_at[5]))
print(' 9. Found {} ({} in split data) coreference(s) at d > 10'.format(corefs_at[6], split_corefs_at[6]))
print(' ---')
print('corefs_at:')
print(corefs_at)
print(' ---')
print('split_corefs_at:')
print(split_corefs_at)
print(' ---')
print('orig_tokens_at:')
print(orig_tokens_at)
print(' ---')
print('split_tokens_at:')
print(split_tokens_at)
#print('10. Split dependencies by name:')
#print(split_dep_by_name)
print(' -----')
print('')

print(' Saving information for Coreference split approach...')
f = open('CorefSplitInfo.data', 'w', encoding='utf-8')
for t in coref_split_info:
    t = [str(i) for i in t]
    f.write(' '.join(t) + "\n")
f.close()

