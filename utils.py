import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def random_data_generator(num):
    result = []
    f = open('./all.txt', 'r')

    data = f.readlines()
    random.shuffle(data)

    for i, line in enumerate(data):
        if i >= num:
            break
        result.append(line[:-1])

    return result


def target_data_generator(target_tag, num):
    result = []
    f = open('./all.txt', 'r')

    data = f.readlines()
    random.shuffle(data)

    for line in data:
        if len(result) >= num:
            break
        if line.split('/')[0] == target_tag:
            result.append(line[:-1])

    return result


def get_confusion_matrix(y_pred, y_true, labels):
    matrix = confusion_matrix(y_pred, y_true)
    if len(matrix) != len(labels):
        raise ValueError('The size of parameter labels[%s] is not equal to the size of labels in matrix[%s]'
                         % (str(len(labels)), str(len(matrix))))
    plt.matshow(matrix, cmap=plt.cm.Blues)
    plt.title('True Label', fontsize=10)
    plt.colorbar()
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.xticks(range(len(matrix)), labels)
    plt.yticks(range(len(matrix)), labels)

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')

    ax.set_ylabel('Predicted label')

    return plt


if __name__ == '__main__':
    get_confusion_matrix([2, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 2], ['ant', 'bee', 'monkey']).show()
