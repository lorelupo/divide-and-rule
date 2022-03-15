# Example of usage:
#
# dir=checkpoints/en-de/iwslt17
# f=logs/large_pronoun.results
# python scripts/mcnemar.py --r1=$dir/fromsplit/k1/$f --r2=$dir/fromsyntsplit/k1/$f --alpha=0.01

import argparse
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar


def main():
    parser = argparse.ArgumentParser(
        description='McNemar’s test to compare two machine learning classifiers.'\
                'The chisquare distribution is used as the approximation to '\
                'the distribution of the test statistic for large sample sizes.'
            )
    # fmt: off
    parser.add_argument(
        '--r1', required=True,
        help='File containing the first column of matched pair results.'
        )
    parser.add_argument(
        '--r2', required=True,
        help='File containing the second column of matched pair results.'
            )
    parser.add_argument(
        '--alpha', default=0.05,
        help="Significance level of the test. Assuming that the null hypothesis is true,"\
            "If the p-value < alpha, we can reject the null hypothesis that the two model's performances are equal."
        )
    # fmt: on

    args = parser.parse_args()

    r1 = pd.read_csv(args.r1, names=['r1'])
    r2 = pd.read_csv(args.r2, names=['r2'])
    df = pd.concat([r1, r2], axis=1)
    contingency_table = sm.stats.Table.from_data(df)
    values = contingency_table.table_orig.values
    if values[0][1] + values[1][0] >= 25:
        # exact=False for using the chisquare distribution as approximation
        # of the distribution of the test statistic (good for large samples)
        # the continuity correction is used
        # ref: http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
        result = mcnemar(values, exact=False, correction=True)
    else:
        print('Outcome differences between the two systems are too small, \
            using an exact binomial test.')
        result = mcnemar(values, exact=True)

    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
    print('statistic=%.3f\np-value=%.3f' % (result.statistic, result.pvalue))


if __name__ == '__main__':
    main()