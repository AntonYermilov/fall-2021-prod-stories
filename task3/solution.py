import numpy as np
from scipy import stats
from argparse import ArgumentParser


def main(src, dst):
    with open(src, 'r') as fp:
        try:
            pairs = list(map(lambda line: list(map(float, line.split(maxsplit=1))), fp.readlines()))
            n, x, y = len(pairs), np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        except:
            raise Exception(f'Invalid data format')
        if n < 9:
            raise Exception(f'Expected at least 9 variables, but found {len(x)}')

    ids = np.argsort(x)
    x, y = x[ids], y[ids]
    r = stats.rankdata(-y)

    p = round(n / 3)
    delta = r[:p].sum() - r[-p:].sum()
    var = (n + 0.5) * np.sqrt(p / 6)
    adj = delta / (p * (n - p))

    with open(dst, 'w') as fp:
        print(f'{round(delta)} {round(var)} {adj:.2f}', file=fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, default='in.txt')
    parser.add_argument('--dst', type=str, default='out.txt')
    args = parser.parse_args()

    main(args.src, args.dst)
