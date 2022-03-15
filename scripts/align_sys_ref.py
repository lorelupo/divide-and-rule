import argparse
import os
import re
import sys

def main():
    """
    Align reference with system output sentences from a concatenation model.
    """
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument('--hyp', help='Path to system hypothesis.')
    parser.add_argument('--ref', help='Path to reference.')
    parser.add_argument('--src', default=None, help='Path to source documents.')
    # fmt: on
    args = parser.parse_args()

    oldsrc = args.src
    oldhyp = args.hyp
    oldref = args.ref
    newsrc = oldsrc + ".new"
    newhyp = oldhyp + ".new" 
    newref = oldref + ".new"

    # newhyp = args.hyp
    # oldhyp = args.hyp + '~'
    # os.rename(newhyp, oldhyp)
    # newref = args.ref
    # oldref = args.ref + '~'
    # os.rename(newref, oldref)

    with  open(oldsrc, 'r') as src, \
            open(oldhyp, 'r') as hyp, \
                open(oldref, 'r') as ref, \
                    open(newsrc, 'w+') as srcout, \
                        open(newhyp, 'w+') as hypout, \
                            open(newref, 'w+') as refout:
        shorter = 0
        longer = 0
        for line in zip(hyp,ref,src):
            if line[0].strip(): # non-empty line. Write it to output
                hypl = re.sub(r'^<\/s> ','',line[0]).split(" </s> ")
                refl = line[1].split(" </s> ")
                srcl = line[2].split(" </s> ")
                if len(hypl) != len(refl):
                    dl = len(refl)-len(hypl)
                    if dl > 0:
                        longer += 1
                        for _ in range(dl):
                            hypl.insert(0,'')
                    elif dl < 0:
                        shorter += 1
                        for _ in range(-dl):
                            del hypl[0]

                assert len(refl)-len(hypl) == 0, \
                    'System hypotesis should match references for comparison'
                srcout.write(str("\n".join(srcl)))
                hypout.write(str("\n".join(hypl)))
                refout.write(str("\n".join(refl)))

        print("#################################")
        print("# Output documents have less sentences than source documents in {} cases, they have more sentences in {} cases.".format(longer,shorter))
        print("#################################")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
