from bisect import bisect_left

def bin_search(x, a, lo=0, hi=None):
    # os.sys.stderr.write(str(x))
    hi = hi if hi is not None else len(a) # hi defaults to len(a)   
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (True if pos != hi and a[pos] == x else False) # don't walk off the end


def open_file(fname):
    if ".gz" in fname: fp = gzip.open(fname, "rt")
    else: fp = open(fname, "r")
    return fp

    
