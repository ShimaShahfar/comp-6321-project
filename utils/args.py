import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        required=True,
        help='The Configuration file')
    args = argparser.parse_args()
    return args
