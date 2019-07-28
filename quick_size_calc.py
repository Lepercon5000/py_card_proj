import math
import itertools
from pprint import PrettyPrinter

PP = PrettyPrinter()


def conv_size(size, kernal_size, stride, padding=0, dilation=1):
    return math.floor(((size + 2 * padding - dilation * (kernal_size - 1) - 1) / stride) + 1)


def conv_trans_size(size, kernal_size, stride, padding=0, output_padding=0):
    return (size - 1) * stride - 2 * padding + kernal_size + output_padding


def reverse_size(input_size, goal_size, depth=3, max_ks=9, max_stride=2):
    pre_experimental_layer = []
    for con_trans_layer_ks in range(max_ks):
        for con_trans_layer_stride in range(max_stride):
            pre_experimental_layer.append(
                (con_trans_layer_ks + 1, con_trans_layer_stride + 1, 0))

    good_results = []
    for combo in itertools.combinations(pre_experimental_layer, depth):
        cur_size = input_size
        for layer in combo:
            cur_size = conv_trans_size(cur_size, layer[0], layer[1], layer[2])
        if cur_size == goal_size:
            good_results.append(combo)

    PP.pprint(good_results)


height = 340 / 2
size = height
print(size)
size = conv_size(size, 1, 1, 0)
print(size)
size = conv_size(size, 9, 2, 0)
print(size)
size = conv_size(size, 2, 2, 0)  # max
print(size)
size = conv_size(size, 3, 1, 0)
print(size)
size = conv_size(size, 2, 2, 0)  # max
print(size)

reverse_size(size, height)

print('next dim')

width = 240 / 2
size = width
print(size)
size = conv_size(size, 9, 2, 0)
print(size)
size = conv_size(size, 2, 2, 0)  # max
print(size)
size = conv_size(size, 3, 1, 0)
print(size)
size = conv_size(size, 2, 2, 0)  # max
print(size)

reverse_size(size, width)
