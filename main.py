from knn_classifier import KnnClassifier

import sys


def main():
    classifier = KnnClassifier(sys.argv[1], sys.argv[2], sys.argv[3])
    classifier.train()

    print('Give vector (after spaces) or write EXIT to close:')

    while True:
        print('Give vector:')
        string = input()

        if string != 'EXIT':
            vector = string.split(' ')
            classifier.classify(vector)
        else:
            break


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main()
    else:
        print('usage: python3 main.py [K] [Train-set path] [Test-set path]')
