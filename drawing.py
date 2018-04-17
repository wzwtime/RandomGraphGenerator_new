# import networkx as nx
import matplotlib.pyplot as plt
import math
"""
G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_nodes_from([3,4,5,6])
G.add_cycle([1,2,3,4])
G.add_edge(1,3)
G.add_edges_from([(3,5),(3,6),(6,7)])
nx.draw(G, with_labels=True)
# plt.savefig("youxiangtu.png")
plt.show()
"""
"""
G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_nodes_from([3,4,5,6])
G.add_cycle([1,2,3,4])
G.add_edge(1,3)
G.add_edges_from([(3,5),(3,6),(6,7)])
nx.draw(G)
# plt.savefig("wuxiangtu.png")
plt.show()


G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (4, 5), (4, 6), (5, 6)])
pos = nx.spring_layout(G)

colors = [1, 2, 2, 2, 3, 1]
nx.draw_networkx_nodes(G, pos, node_color=colors)
nx.draw_networkx_edges(G, pos)

plt.axis('off')
# plt.savefig("color_nodes.png")
plt.show()
"""

SLR_heft = {
    20: [],
    40: [],
    60: [],
    80: [],
    100: [],
}
SLR_cpop = {
    20: [],
    40: [],
    60: [],
    80: [],
    100: [],
}


Speedup_heft = {
    20: [],
    40: [],
    60: [],
    80: [],
    100: [],
}
Speedup_cpop = {
    20: [],
    40: [],
    60: [],
    80: [],
    100: [],
}


def read_slr_speedup(slr_name, speedup_name):
    """read in slr file"""
    filename = "slr_speedup_heft_cpop.txt"
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line_list = line.split()
            v_ = int(line_list[0])
            if slr_name == SLR_heft and speedup_name == Speedup_heft:
                heft_slr_ = float(line_list[5])
                slr_name[v_].append(heft_slr_)
                speedup_heft_ = float(line_list[7])
                speedup_name[v_].append(speedup_heft_)

            elif slr_name == SLR_cpop and speedup_name == Speedup_cpop:
                cpop_slr_ = float(line_list[6])
                slr_name[v_].append(cpop_slr_)
                speedup_cpop_ = float(line_list[8])
                speedup_name[v_].append(speedup_cpop_)


read_slr_speedup(SLR_heft, Speedup_heft)
read_slr_speedup(SLR_cpop, Speedup_cpop)

print("SLR_heft=", SLR_heft)
print("SLR_cpop=", SLR_cpop)
print("speedup_heft=", Speedup_heft)
print("speedup_cpop=", Speedup_cpop)

heft_info = []
cpop_info = []


def avg_slr(slr_list, info):
    """computing avg slr"""
    for key, values in slr_list.items():
        # print(key, values)
        temp_info = []
        min_info = min(values)
        temp_info.append(min_info)
        max_info = max(values)
        temp_info.append(max_info)
        avg_slr_ = round(sum(values) / len(values), 2)
        temp_info.append(avg_slr_)
        info.append(temp_info)


# avg_slr(SLR_heft, heft_info)
# print("heft- avg_slr =", heft_info)
#
# avg_slr(SLR_cpop, cpop_info)
# print("cpop- avg_slr =", cpop_info)
#

def avg_speedup(speedup_list, info):
    """computing avg slr"""
    for key, values in speedup_list.items():
        # print(key, values)
        temp_info = []
        min_info = min(values)
        temp_info.append(min_info)
        max_info = max(values)
        temp_info.append(max_info)
        avg_speedup_ = round(sum(values) / len(values), 2)
        temp_info.append(avg_speedup_)
        info.append(temp_info)


avg_speedup(Speedup_heft, heft_info)
print("heft-avg_speedup =", heft_info)

avg_speedup(Speedup_cpop, cpop_info)
print("cpop-avg_speedup =", cpop_info)


def drawing_avg_slr():
    """绘制折线图"""
    input_values = list(range(20, 101, 20))   # 任务数量
    # plt.title("Average Job Slowdown of different scheduling algorithms.")
    plt.xlabel("Number of Nodes", fontsize=11)
    plt.ylabel("Average SLR", fontsize=11)
    # my_x_ticks = range(20, 101, 20)
    # plt.xticks(my_x_ticks)

    # plt.xlim(0, int(length) + 1)
    # plt.ylim(2, 5)
    y1_value = []
    y2_value = []
    for i in range(len(heft_info)):
        y1_value.append(heft_info[i][2])
    for i in range(len(cpop_info)):
        y2_value.append(cpop_info[i][2])
    plt.plot(input_values, y1_value, "^-.", label="HEFT")
    plt.plot(input_values, y2_value, "8--", label="CPOP")

    # plt.grid(True)         # 显示网格
    plt.legend(loc='upper left')           # 显示图例
    # plt.savefig('asd.png', bbox_inches='tight')
    # plt.close()
    plt.show()


# drawing_avg_slr()

# for key, values in SLR_heft.items():
#     y1 = tuple(values)
#     # print(tuple(y1))


def drawing_avg_speedup():
    """绘制ASD折线图"""
    input_values = list(range(20, 101, 20))   # 任务数量
    # plt.title("Average Job Slowdown of different scheduling algorithms.")
    plt.xlabel("Number of Nodes", fontsize=11)
    plt.ylabel("Average Speedup", fontsize=11)
    # my_x_ticks = range(20, 101, 20)
    # plt.xticks(my_x_ticks)

    # plt.xlim(0, int(length) + 1)
    # plt.ylim(2, 5)
    y1_value = []
    y2_value = []
    for i in range(len(heft_info)):
        y1_value.append(heft_info[i][2])
    for i in range(len(cpop_info)):
        y2_value.append(cpop_info[i][2])
    plt.plot(input_values, y1_value, "^-.", label="HEFT")
    plt.plot(input_values, y2_value, "8--", label="CPOP")

    # plt.grid(True)         # 显示网格
    plt.legend(loc='upper left')           # 显示图例
    # plt.savefig('asd.png', bbox_inches='tight')
    # plt.close()
    plt.show()


# drawing_avg_speedup()


"""slr_box"""
"""
plt.title("HEFT")
plt.xlabel("Number of Nodes", fontsize=11)
plt.ylabel("Schedule Length Ratio", fontsize=11)
plt.boxplot(tuple(SLR_heft.values()), labels=SLR_heft.keys())
plt.show()
"""

"""speedup_box"""

plt.title("CPOP")
plt.xlabel("Number of Nodes", fontsize=11)
plt.ylabel("Speedup", fontsize=11)
plt.boxplot(tuple(Speedup_cpop.values()), labels=Speedup_cpop.keys())
plt.show()


slr_ccr_heft = {
    0.1: [],
    0.5: [],
    1.0: [],
    5.0: [],
    10.0: [],
}
slr_ccr_cpop = {
    0.1: [],
    0.5: [],
    1.0: [],
    5.0: [],
    10.0: [],
}


def read_slr_ccr(slr_ccr_name):
    """read in slr_ccr_name file"""
    filename = "ccr.txt"
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line_list = line.split()
            ccr_ = float(line_list[0])
            if slr_ccr_name == slr_ccr_heft:
                heft_slr_ = float(line_list[5])
                slr_ccr_name[ccr_].append(heft_slr_)
            elif slr_ccr_name == slr_ccr_cpop:
                cpop_slr_ = float(line_list[6])
                slr_ccr_name[ccr_].append(cpop_slr_)


# read_slr_ccr(slr_ccr_heft)
#
# print("heft-slr_ccr=", slr_ccr_heft)
# read_slr_ccr(slr_ccr_cpop)
#
# print("cpop-slr_ccr=", slr_ccr_cpop)


"""slr_ccr_box"""
"""
plt.title("HEFT")
plt.xlabel("CCR", fontsize=11)
plt.ylabel("Schedule Length Ratio", fontsize=11)
plt.boxplot(tuple(slr_ccr_heft.values()), labels=slr_ccr_cpop.keys())
plt.show()

"""
avg_slr_ccr_heft = []
avg_slr_ccr_cpop = []


def avg_slr_ccr(slr_ccr_name, info):
    """computing avg slr_ccr"""
    for key, values in slr_ccr_name.items():
        # print(key, values)
        avg_slr_ = round(sum(values) / len(values), 2)
        info.append(avg_slr_)


# avg_slr_ccr(slr_ccr_heft, avg_slr_ccr_heft)
# print("heft-avg_slr_ccr=", avg_slr_ccr_heft)
# avg_slr_ccr(slr_ccr_cpop, avg_slr_ccr_cpop)
# print("cpop-avg_slr_ccr=", avg_slr_ccr_cpop)
#

def drawing_avg_slr_ccr():
    """绘制折线图"""
    input_values = (1, 2, 3, 4, 5)  # 任务数量
    # input_values = (0.1, 0.5, 1.0, 5.0, 10.0)  # 任务数量
    # plt.title("Average Job Slowdown of different scheduling algorithms.")
    plt.xlabel("CCR", fontsize=11)
    plt.ylabel("Average SLR", fontsize=11)
    # my_x_ticks = range(20, 101, 20)
    # plt.xticks(my_x_ticks)

    # plt.xlim(0, int(length) + 1)
    # plt.ylim(2, 5)
    y1_value = tuple(avg_slr_ccr_heft)
    y2_value = tuple(avg_slr_ccr_cpop)

    plt.plot(input_values, y1_value, "^-.", label="HEFT")
    plt.plot(input_values, y2_value, "8--", label="CPOP")

    # plt.grid(True)         # 显示网格
    plt.legend(loc='upper left')           # 显示图例
    # plt.savefig('asd.png', bbox_inches='tight')
    # plt.close()
    plt.show()


# drawing_avg_slr_ccr()


slr_beta_heft = {
    0.1: [],
    0.25: [],
    0.5: [],
    0.75: [],
    1.0: [],
}
slr_beta_cpop = {
    0.1: [],
    0.25: [],
    0.5: [],
    0.75: [],
    1.0: [],
}


def read_slr_beta(slr_beta_name):
    """read in slr_ccr_name file"""
    filename = "slr_beta.txt"
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line_list = line.split()
            beta_ = float(line_list[0])
            if slr_beta_name == slr_beta_heft:
                heft_slr_ = float(line_list[5])
                slr_beta_name[beta_].append(heft_slr_)
            elif slr_beta_name == slr_beta_cpop:
                cpop_slr_ = float(line_list[6])
                slr_beta_name[beta_].append(cpop_slr_)

#
# read_slr_beta(slr_beta_heft)
# print("heft=", slr_beta_heft)
#
# read_slr_beta(slr_beta_cpop)
# print("cpop=", slr_beta_cpop)

"""slr_beta_box"""
"""
plt.title("CPOP")
plt.xlabel("Beta", fontsize=11)
plt.ylabel("Schedule Length Ratio", fontsize=11)
plt.boxplot(tuple(slr_beta_cpop.values()), labels=slr_beta_cpop.keys())
plt.show()
"""

avg_slr_beta_heft = []
avg_slr_beta_cpop = []


def avg_slr_beta(slr_beta_name, info):
    """computing avg slr_beta"""
    for key, values in slr_beta_name.items():
        # print(key, values)
        avg_slr_ = round(sum(values) / len(values), 2)
        info.append(avg_slr_)


# avg_slr_beta(slr_beta_heft, avg_slr_beta_heft)
# print("heft-avg_slr_beta=", avg_slr_beta_heft)
# avg_slr_beta(slr_beta_cpop, avg_slr_beta_cpop)
# print("cpop-avg_slr_beta=", avg_slr_beta_cpop)


def drawing_avg_slr_beta():
    """绘制折线图"""
    # input_values = (0.1, 0.5, 1.0, 5.0, 10.0)  # 任务数量
    input_values = (0.1, 0.25, 0.50, 0.75, 1.0)  # 任务数量
    # plt.title("Average Job Slowdown of different scheduling algorithms.")
    plt.xlabel("BETA", fontsize=11)
    plt.ylabel("Average SLR", fontsize=11)
    # my_x_ticks = range(20, 101, 20)
    # plt.xticks(my_x_ticks)

    # plt.xlim(0, int(length) + 1)
    # plt.ylim(2, 5)
    y1_value = tuple(avg_slr_beta_heft)
    y2_value = tuple(avg_slr_beta_cpop)

    plt.plot(input_values, y1_value, "^-.", label="HEFT")
    plt.plot(input_values, y2_value, "8--", label="CPOP")

    # plt.grid(True)         # 显示网格
    plt.legend(loc='best')           # 显示图例
    # plt.savefig('asd.png', bbox_inches='tight')
    # plt.close()
    plt.show()


# drawing_avg_slr_beta()
