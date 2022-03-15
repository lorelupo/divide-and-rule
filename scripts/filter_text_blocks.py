import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Split text in blocks of B lines \
            and keep a block every E blocks, starting from block S',
    )
    # fmt: off
    parser.add_argument('in_path',help='Input dataset file paths.')
    parser.add_argument('out_path',help='Output dataset file paths.')
    parser.add_argument('--B', type=int, default=None,
        help='Number of lines in a block'
    )
    parser.add_argument('--E', type=int, default=2,
        help='Keep a block every E blocks'
    )
    parser.add_argument('--S', type=int, default=0,
        help='Retain every E blocks starting from the S one'
    )

    # fmt: on
    args = parser.parse_args()

    with open(args.in_path, 'r') as fin, open(args.out_path, 'w') as fout:
        lines = np.array(fin.readlines())
        ll = len(lines)
        slines = np.split(lines, list(range(args.B, ll, args.B)))
        fout.write(''.join(list(np.concatenate(slines[args.S::args.E]))))

if __name__ == '__main__':
    main()