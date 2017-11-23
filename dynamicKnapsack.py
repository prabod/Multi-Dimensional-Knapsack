import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser()

parser.add_argument('-f', action='store', dest='file_name',
                    help='Enter the File Name')

results = parser.parse_args()


def write_to_file(output):
    with open(results.file_name + '_out_DP', 'w') as input_file:
        val = copy.deepcopy(values)
        for i in range(len(val)):
            if val[i] in output:
                input_file.write('1\n')
                output.remove(val[i])
            else:
                input_file.write('0\n')


def subset_sum(numbers, target, partial=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target:
        print "%s %s" % (partial, target)
        write_to_file(partial)

    if s >= target:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i + 1:]
        subset_sum(remaining, target, partial + [n])


def knapsack(M, N, values, weights, bounds):
    shape = (M,) + tuple(np.array(bounds) + 1)
    table = np.negative(np.ones(shape=shape, dtype=np.int))
    maxVal = f(M - 1, table, values, weights, np.array(bounds))
    subset_sum(values, maxVal)


"""
Recursive function definition for the dynamic programming approach
"""


def f(j, table, val, wei, bound):
    b = [[value] for (x), value in np.ndenumerate(bound)]
    if j == -1:
        return 0
    elif np.any(np.array(bound) == 0):
        return 0

    elif table[[j] + b] != -1:
        return table[[j] + b]
    else:
        if np.all((np.array(bound) - np.array(weights[j]) >= 0)):
            dum = [val[j] + int(f(j - 1, table, val, wei, np.array(bound) - np.array(weights[j]))),
                   int(f(j - 1, table, val, wei, np.array(bound)))]
        else:
            dum = [int(f(j - 1, table, val, wei, np.array(bound)))]
        table[[j] + b] = max(dum)
        return max(dum)

"""
Read File Inputs
"""


def get_file_input(file_name):
    f = open(file_name, 'r')
    M = int(f.readline())
    N = int(f.readline())
    values = []
    weights = []
    for i in range(M):
        line = f.readline()
        line = map(int, line.split(' '))
        values.append(line[0])
        weights.append(line[1:])
    bounds = map(int, f.readline().strip().split(' '))

    return M, N, values, weights, bounds


M, N, values, weights, bounds = get_file_input(results.file_name)
knapsack(M, N, values, weights, bounds)
