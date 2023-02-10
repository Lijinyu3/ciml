from sklearn.tree import *
from data import *
import matplotlib.pyplot as plt
import matplotlib.ticker as tker


def all_depth_array(start: int, end: int, loss_func, data_files):
    """
    >>> data = [r'data/sentiment.tr']
    >>> loss_func = lambda f, X, Y: np.mean(f.predict(X) == Y)
    >>> len(all_depth_array(1, 3, loss_func, data))
    1
    """
    loss_data = []
    all_data = read_all_data(data_files)
    tree_l = [create_train_tree(i, all_data[0]) for i in range(start, end + 1)]
    for d in all_data:
        loss_data.append(get_error(d, tree_l, loss_func, start, end))
    return loss_data

def get_error(data, f, loss_func, start, end):
    loss = []
    for i in range(start, end + 1):
        loss.append(loss_func(f[i - 1], data[0], data[1]))
    return loss


def read_all_data(data_url: list) -> list:
    """
    >>> sufix = ['tr', 'de', 'te']
    >>> data_url = [r'data/sentiment.' + i for i in sufix]
    >>> d = read_all_data(data_url)
    >>> len(d)
    3
    >>> len(d[0])
    3
    """

    data_list = []
    dictionary = None
    for f in data_url:
        data_list.append(loadTextDataBinary(f, dictionary))
        dictionary = data_list[-1][-1]

    return data_list


def create_train_tree(depth: int, train_data):
    """
    >>> data = read_all_data([r'data/sentiment.tr'])
    >>> t = create_train_tree(2, data[0])
    >>> val = np.mean(t.predict(data[0][0]) == data[0][1])
    >>> assert isinstance(val, float)
    """
    X, Y = train_data[0], train_data[1]
    dt = DecisionTreeClassifier(max_depth= depth)
    dt.fit(X, Y)
    return dt


if __name__ == '__main__':
   data_prefix = r'data/sentiment.'
   data_sufix = ('tr', 'de', 'te')

   urls = [data_prefix + su for su in data_sufix]
   loss_func = lambda f, X, Y: np.mean(f.predict(X) == Y)
   start, end = 1, 20

   data = all_depth_array(start, end, loss_func, urls)

   fig, ax = plt.subplots(layout='constrained')
   x = range(start, end + 1)
   ax.plot(x, data[0], label='train')
   ax.plot(x, data[1], label='dev')
   ax.plot(x, data[2], label='test')
   plt.xticks(range(start, end + 1))
   plt.xlabel('depth')
   plt.ylabel('expected loss')
   plt.legend()
   plt.show()
