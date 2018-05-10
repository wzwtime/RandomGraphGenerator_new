import os

dag = {}


def read_dag(q, n):
    """q is the number of processors, n is which graph"""
    filename = 'save_dag\q=' + str(q) + '\_' + str(n) + '_dag_q=' + str(q) + '.txt'
    """
    f = open(filename)
    last_line = ''.join(f.readlines()[-1])
    line_list = last_line.split()
    v = int(line_list[1])
    print(v)
    count = len(open(filename, 'rU').readlines())
    print(count)
    """
    with open(filename, 'r') as file_object_:
        lines = file_object_.readlines()
        task_id = 0
        for line in lines:
            line_list = line.split()
            task_id = int(line_list[0])
            succ_dict = {}
            for line_ in lines:
                line_list_ = line_.split()
                task_id_ = int(line_list_[0])
                succ_id_ = int(line_list_[1])
                succ_weight = int(line_list_[2])
                if task_id == task_id_:
                    succ_dict[succ_id_] = succ_weight
            dag[task_id] = succ_dict
        dag[task_id + 1] = {}


# read_dag(3, 1)

computation_costs = []


def read_computation_costs(q, n):
    """read computation_costs.txt"""
    filename = 'save_dag\q=' + str(q) + '\_' + str(n) + '_computation_costs_q=' + str(q) + '.txt'
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            line_list = line.split()
            temp_list = []
            for i in range(len(line_list)):
                temp_list.append(int(line_list[i]))
            computation_costs.append(temp_list)
    return computation_costs


# read_computation_costs(3, 1)
