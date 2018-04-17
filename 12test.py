import operator

dag = {
    1: {2: 18, 3: 12, 4: 9, 5: 11, 6: 14},
    2: {8: 19, 9: 16},
    3: {7: 23},
    4: {8: 27, 9: 23},
    5: {9: 13},
    6: {8: 15},
    7: {10: 17},
    8: {10: 11},
    9: {10: 13},
    10: {},
}

dag1 = sorted(dag.items(), key=operator.itemgetter(0))  # 按任务编号升序排序
w = {1: {2: 19, 3: 5, 4: 10}, 2: {5: 10}, 3: {6: 10, 7: 5}, 4: {8: 4}, 5: {9: 19}, 6: {10: 4, 11: 12, 12: 9}, 7: {13: 5}, 8: {14: 7}, 9: {15: 4}, 10: {16: 17}, 11: {17: 14}, 12: {18: 21}, 13: {19: 2}, 14: {20: 2}, 15: {21: 7}, 16: {22: 4}, 17: {24: 18, 23: 23}, 18: {25: 5}, 19: {26: 5}, 20: {27: 15}, 21: {28: 1}, 22: {29: 22}, 23: {29: 23}, 24: {30: 5}, 25: {31: 8}, 26: {32: 13}, 27: {33: 5}, 28: {34: 22}, 29: {35: 22}, 30: {36: 14}, 31: {37: 20}, 32: {38: 11}, 33: {39: 15}, 34: {40: 4}, 35: {41: 13}, 36: {42: 2}, 37: {43: 23}, 38: {43: 13}, 39: {43: 17}, 40: {44: 19}, 41: {45: 12}, 42: {48: 10, 49: 12, 46: 21, 47: 3}, 43: {50: 6}, 44: {51: 18}, 45: {52: 8}, 46: {53: 21}, 47: {54: 15}, 48: {55: 17}, 49: {56: 7}, 50: {56: 12}, 51: {57: 15}, 52: {58: 5}, 53: {58: 18}, 54: {59: 7}, 55: {60: 21}, 56: {61: 5}, 57: {62: 7}, 58: {64: 17, 63: 5}, 59: {65: 19}, 60: {66: 23}, 61: {67: 14, 68: 19}, 62: {69: 15}, 63: {70: 13}, 64: {71: 14}, 65: {72: 10}, 66: {73: 8}, 67: {74: 11}, 68: {75: 10}, 69: {76: 21}, 70: {77: 11}, 71: {78: 19}, 72: {79: 17}, 73: {80: 18}, 74: {81: 17}, 75: {82: 8}, 76: {83: 18}, 77: {83: 19}, 78: {84: 5}, 79: {84: 3}, 80: {84: 9}, 81: {85: 16}, 82: {86: 5}, 83: {87: 6}, 84: {88: 14}, 85: {89: 11, 90: 7, 91: 5}, 86: {92: 12}, 87: {93: 3}, 88: {94: 10}, 89: {95: 22}, 90: {96: 3}, 91: {97: 17, 98: 10}, 92: {99: 13}, 93: {100: 1}, 94: {100: 5}, 95: {100: 22}, 96: {100: 13}, 97: {100: 19}, 98: {100: 9}, 99: {100: 13}, 100: {}}
print(len(w))
