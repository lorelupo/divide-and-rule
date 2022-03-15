import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Take as input a text file and keep only alternate blocks of n lines '
    )
    # fmt: off
    parser.add_argument('input_path',help='Input dataset file paths.')
    parser.add_argument(
        '--block-size',
        type=int,
        metavar='N',
        default=4,
        help='how many consecutive lines constitute a block'
    )
    # fmt: on
    args = parser.parse_args()
    
    N = args.block_size
    n = 0
    with open(args.input_path, 'r') as infile:
        with open(args.input_path + ".block", 'w+') as outfile:
            for line in infile:
                if 0 <= n < N:
                    outfile.write(line)
                    n += 1
                else:
                    n += 1
                    if n == 2*N:
                        n = 0

if __name__ == '__main__':
    main()