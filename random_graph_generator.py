# coding=utf-8
import operator
import random
import math
import os


"""random_graph_generator2018-03-26"""
SET_v = [20, 40, 60, 80, 100]
SET_ccr = [0.1, 0.5, 1.0, 5.0, 10.0]
SET_alpha = [1.0, 2.0]
SET_out_degree = [2, 3, 4, 5, ]
SET_beta = [0.1, 0.25, 0.5, 0.75, 1.0]
computation_costs = []
dag = {}
new_dag = {}


def random_avg_w_dag(n_min, n_max):
    """Randomly generated average computation costs"""
    avg_w_dag_ = random.randint(n_min, n_max)
    return avg_w_dag_


# Randomly generated average computation costs
avg_w_dag = random_avg_w_dag(5, 20)


def get_wij(v, p, beta):
    """Generate computation costs overhead for tasks on different processors"""
    filename = 'computation_costs.txt'
    if os.path.exists(filename):
        os.remove(filename)  # remove
    for i in range(v):
        avg_w = random.randint(1, 2 * avg_w_dag)
        temp_list = []
        for j in range(p):
            wij = random.randint(math.ceil(avg_w * (1 - beta / 2)), math.ceil(avg_w * (1 + beta / 2)))
            temp_list.append(wij)
        computation_costs.append(temp_list)
        with open(filename, 'a') as file_object2:
            info1 = str(temp_list) + "\n"
            file_object2.write(info1)


def get_height_width(v, alpha):
    """get_height_width"""
    mean_height = math.ceil(math.sqrt(v) / alpha)       # Round up and calculate the average
    mean_width = math.ceil(alpha * math.sqrt(v))        # Round up and calculate the average
    height = random.randint(1, 2 * mean_height - 1)     # uniform distribution with a mean value equal to mean_height
    width = random.randint(2, 2 * mean_width - 2)       # uniform distribution with a mean value equal to mean_width
    return height, width


def number_nodes_layer(height, width, sum_m, num_second_layer):
    """determine the number of nodes in each layer of the graph"""
    task_num_layer = []
    for t in range(height - 4):
        task_num_layer.append(2)
    for k in range(sum_m - 2 * (height - 4)):
        rand_index = random.randint(0, height - 5)
        if task_num_layer[rand_index] < width:
            task_num_layer[rand_index] += 1
        else:
            min_n = min(task_num_layer)
            min_index = task_num_layer.index(min_n)
            task_num_layer[min_index] += 1
    task_num_layer.insert(0, 1)                     # the first layer
    task_num_layer.insert(1, num_second_layer)      # the second layer
    task_num_layer.insert(int(height / 2), width)   # width
    task_num_layer.append(1)                        # the last layer
    return task_num_layer


def order_dag(height, task_num_layer, out_degree):
    """Order the number of nodes per layer according to the out-degree"""
    for j in range(height - 1):
        for i in range(height - 1):
            if task_num_layer[i] * out_degree < task_num_layer[i + 1]:
                temp = task_num_layer[i]
                task_num_layer[i] = task_num_layer[i + 1]
                task_num_layer[i + 1] = temp
                # task_num_layer[i], task_num_layer[i + 1] = task_num_layer[i + 1], task_num_layer[i]   # don't use temp
    return task_num_layer


def get_dag_id(height, task_num_layer):
    """Convert the number of nodes per layer to sequential task numbers."""
    dag_id = []
    num = 0
    for i in range(height):
        dag_id_temp = []
        for j in range(int(task_num_layer[i])):
            num += 1
            dag_id_temp.append(num)
        dag_id.append(dag_id_temp)
    return dag_id


def the_first_layer(dag_id, avg_comm_costs):
    """the first layer"""
    temp_dag = {}
    for i in range(len(dag_id[1])):  # the first layer
        index = dag_id[1][i]
        communication_costs = random.randint(1, 2 * avg_comm_costs - 1)
        temp_dag[index] = communication_costs
    dag[1] = temp_dag


def second_to_last_layer(dag_id, height, avg_comm_costs):
    """Second-to-last layer"""
    for i in range(len(dag_id[height - 2])):
        temp_dag = {}                           # Attention!!!!!!!!!!! Prevents generator the same communication costs
        index = dag_id[height - 2][i]
        dag_index = dag_id[height - 1][0]
        communication_costs = random.randint(1, 2 * avg_comm_costs - 1)
        temp_dag[dag_index] = communication_costs
        dag[index] = temp_dag


def grouping_children_nodes(p_num, c_num, out_degree):
    """Grouping children's nodes"""
    temp_child_num = []
    for j in range(p_num):
        temp_child_num.append(1)
    for k in range(c_num - p_num):
        rand_index = random.randint(0, p_num - 1)
        if temp_child_num[rand_index] < out_degree:
            temp_child_num[rand_index] += 1
        else:
            min_n = min(temp_child_num)
            min_index = temp_child_num.index(min_n)
            temp_child_num[min_index] += 1
    return temp_child_num


def less_to_multi(task_num_layer, out_degree, dag_id, avg_comm_costs):
    """Less-to-multi make child nodes are randomly divided into total number of parent nodes"""
    for i in range(1, len(task_num_layer) - 2):
        p_num = task_num_layer[i]
        c_num = task_num_layer[i + 1]
        if p_num != 1 and c_num != 1 and p_num <= c_num:
            p_index = i
            """Grouping children's nodes"""
            temp_child_num = grouping_children_nodes(p_num, c_num, out_degree)

            """Traversing every parent node of the index."""
            sum_num = 0
            sum_list = 0
            for j in range(p_num):
                temp_dag = {}
                p_id = dag_id[p_index][j]
                """Determination of child node number."""
                """
                sum_num = p_id + p_num - j - 1    # The last parent node number
                print("last_parent_id = ", sum_num)
                """
                if j > 0:
                    sum_list += temp_child_num[j - 1]
                """View subnode number"""
                for k in range(temp_child_num[j]):
                    if j == 0:
                        sum_num = p_id + p_num - j - 1 + k + 1
                    elif j > 0:
                        sum_num = p_id + p_num - j - 1 + k + 1 + sum_list
                    """assign communication costs"""
                    communication_costs = random.randint(1, 2 * avg_comm_costs - 1)
                    temp_dag[sum_num] = communication_costs
                dag[p_id] = temp_dag


def grouping_parent_nodes(p_num, c_num):
    """Grouping children's nodes"""
    temp_parent_num = []
    for j in range(c_num):
        temp_parent_num.append(1)
    for k in range(p_num - c_num):
        rand_index = random.randint(0, c_num - 1)
        temp_parent_num[rand_index] += 1
    return temp_parent_num


def multi_to_less(task_num_layer, dag_id, avg_comm_costs):
    """Multi-to-Less make parent nodes are randomly divided into total number of child nodes"""
    for i in range(2, len(task_num_layer) - 1):     # !!!!!!!!!!!!!  is -1 not -2  Traverse completely
        p_num = task_num_layer[i - 1]
        c_num = task_num_layer[i]
        if p_num != 1 and c_num != 1 and p_num > c_num:
            c_index = i
            """The parent node is randomly divided into the total number of child nodes."""
            temp_parent_num = grouping_parent_nodes(p_num, c_num)

            """Traversing every child node of the index."""
            length_parent = 0
            for j in range(c_num):
                c_id = dag_id[c_index][j]
                first_parent_id = c_id - p_num
                """View parent node id"""
                for k in range(temp_parent_num[j]):
                    length_parent += 1
                    p_id = first_parent_id + length_parent - j - 1
                    """assign communication costs"""
                    temp_dag = {}               # Attention!!!!!!!!!!! Prevents generator the same communication costs
                    communication_costs = random.randint(1, 2 * avg_comm_costs - 1)
                    temp_dag[c_id] = communication_costs
                    dag[p_id] = temp_dag


def random_graph_generator(v, ccr, alpha, out_degree, beta, p):
    """requires five parameters to build weighted DAGs
    v: number of tasks in the graph
    ccr: average communication cost to average computation cost
    alpha: shape parameter of the graph
    out_degree: out degree of a node
    beta: range percentage of computation costs on processors
    p: number of processors"""

    """Determine whether it can constitute a DAG"""
    mean_height = math.ceil(math.sqrt(v) / alpha)  # Round up and calculate the average
    mean_width = math.ceil(alpha * math.sqrt(v))  # Round up and calculate the average
    height = random.randint(1, 2 * mean_height - 1)  # uniform distribution with a mean value equal to mean_height
    width = random.randint(2, 2 * mean_width - 2)  # uniform distribution with a mean value equal to mean_width
    min_num = min(width, out_degree)
    while True:
        num_second_layer = random.randint(2, min_num)
        sum_m = v - 2 - num_second_layer - width
        # print("v =", v, "height = ", height, "width =", width)
        if (height - 4) * width >= sum_m and (2 * (height - 4) <= sum_m):
            print("yes")
            break
        else:
            height = get_height_width(v, alpha)[0]      # random generator a new h,w
            width = get_height_width(v, alpha)[1]
            while (height - 2) * width < v - 2:
                height = get_height_width(v, alpha)[0]  # random generator a new h,w
                width = get_height_width(v, alpha)[1]

    """ 1) The first is to determine the number of nodes in each layer of the graph"""
    task_num_layer = number_nodes_layer(height, width, sum_m, num_second_layer)

    """Order the number of nodes per layer according to the out-degree"""
    task_num_layer = order_dag(height, task_num_layer, out_degree)

    """Convert the number of nodes per layer to sequential task numbers."""
    dag_id = get_dag_id(height, task_num_layer)

    """If there is one node of the dag's first layer,it's a truly dag."""
    if task_num_layer[0] == 1:
        print("v =", v, "height = ", height, "width =", width, "CCR =", ccr, "Alpha =", alpha,
              "out_degree =", out_degree, "beta =", beta, "Number of Processors =", p)
        print("ordered task_num_layer:", task_num_layer)
        print("dag_id = ", dag_id)

        """Generate computation costs on different processors for every task"""
        get_wij(v, p, beta)

        """Average communication costs"""
        avg_comm_costs = math.ceil(ccr * avg_w_dag)  # Rounded up

        """2)Then according to the out-degree to determine the vertex connection relationship,
        allocation of communication costs"""
        """1.the first layer"""
        the_first_layer(dag_id, avg_comm_costs)

        """2.Second-to-last layer"""
        second_to_last_layer(dag_id, height, avg_comm_costs)

        """3.Other layers that remove the last layer"""

        """3.1 Less-to-multi make child nodes are randomly divided into total number of parent nodes"""
        less_to_multi(task_num_layer, out_degree, dag_id, avg_comm_costs)

        """3.2 Multi-to-Less make parent nodes are randomly divided into total number of child nodes"""
        multi_to_less(task_num_layer, dag_id, avg_comm_costs)

        """4.The last layer"""
        dag[v] = {}
    else:
        print("DAG Error! Get a new DAG!")
        print(v)
        random_graph_generator(v, ccr, alpha, 5, beta, 3)      # Get a new DAG


def random_index(set_):
    """Get the random index i of the collection to determine which parameter in the collection"""
    length = len(set_)
    index_ = random.randint(1, length) - 1
    return index_


def select_parameter():
    """Select 5 parameters"""
    v = SET_v[random_index(SET_v)]
    ccr = SET_ccr[random_index(SET_ccr)]
    alpha = SET_alpha[random_index(SET_alpha)]
    # out_degree = SET_out_degree[random_index(SET_out_degree)]
    beta = SET_beta[random_index(SET_beta)]
    # p = random.randint(2, 5)    # processors
    random_graph_generator(20, ccr, alpha, 5, beta, 3)
    """ 
    # Write to file
    filename = 'graph_parameter.txt'
    with open(filename, 'w') as file_object:
        info = str(v) + "  " + str(ccr) + "  " + str(alpha) + "  " + str(out_degree) + "  " + str(beta) + "\n"
        file_object.write(info)
    random_graph_generator(v, ccr, alpha, 5, beta, p)
    """


select_parameter()
dag1 = sorted(dag.items(), key=operator.itemgetter(0))  # Ascending sort by task number
print("dag =", dag1)
# print("computation_costs =", computation_costs)


# Store DAG  in files
filename_ = 'dag.txt'
if os.path.exists(filename_):
    os.remove(filename_)  # remove
for m in range(len(dag1)):
    task_id = dag1[m][0]
    for key, value in dag1[m][1].items():
        succ_id = key
        succ_weight = value
        filename_ = 'dag.txt'   # build a new file
        with open(filename_, 'a') as file_object:
            info = str(task_id) + " " + str(succ_id) + " " + str(succ_weight) + "\n"
            file_object.write(info)


def read_dag():
    """read the dat.txt build dag"""
    filename = 'dag.txt'
    if os.path.exists(filename):  # dag.txt exists
        with open(filename, 'r') as file_object_:
            lines = file_object_.readlines()
            task_id_ = 0
            for n in range(len(dag1)):
                succ_dict = {}
                for line in lines:
                    line_list = line.split()
                    task_id1 = int(line_list[0])
                    succ_id_ = int(line_list[1])
                    succ_weight_ = int(line_list[2])

                    if task_id1 == int(dag1[n][0]):
                        task_id_ = task_id1
                        succ_dict[succ_id_] = succ_weight_
                if task_id_ == int(dag1[n][0]):
                    new_dag[task_id_] = succ_dict
        last_task_id = dag1[-1][0]
        new_dag[last_task_id] = {}


read_dag()


def read_computation_costs(v):
    """read computation_costs.txt"""
    filename = 'computation_costs.txt'
    temp_list = []
    if os.path.exists(filename):  # computation_costs exists
        with open(filename, 'r') as file_object3:
            lines = file_object3.readlines()
            for i in range(v):
                temp_list.append(lines)
    return temp_list
