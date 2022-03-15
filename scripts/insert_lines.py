import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Take as input 2 text files and merge them by inserting \
            a line from file2 every N lines of file1.',
    )
    # fmt: off
    parser.add_argument('file1', help='First input text file.')
    parser.add_argument('file2', help='Second input text file.')
    parser.add_argument('outfile', help='Output file.')
    parser.add_argument('N', type=int,
    help='Insert a line from file2 every N lines of file1.'
    )
    # fmt: on
    args = parser.parse_args()

    with open(args.file1, 'r') as f1:
        with open(args.file2, 'r') as f2:
            with open(args.outfile, 'w+') as of:
                for n, l1 in enumerate(f1):
                    of.write(l1)
                    if (n+1)%(args.N) == 0:
                        l2 = f2.readline()
                        of.write(l2)

if __name__ == '__main__':
    main()