# coding=utf
"""The HEFT Scheduling Algorithm of DAGs"""
import operator
import random_graph_generator as rgg
import copy

M = 10000
# v = 100
v = len(rgg.new_dag)
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
eft = 0             # Recording the make_span
scheduler = []      # Recording the scheduling procedures'id of different tasks.
avg_costs = []      # Computing the average computation costs of every task.
rank_u = []         # Recording the priorities
pred = []       # Predecessor node list

dag = rgg.new_dag

computation_costs = rgg.computation_costs

"""
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
"""
"""
dag = {
    1: {2: 17, 3: 31, 4: 29, 5: 13, 6: 7},
    2: {8: 3, 9: 30},
    3: {7: 16},
    4: {8: 11, 9: 7},
    5: {9: 57},
    6: {8: 5},
    7: {10: 9},
    8: {10: 42},
    9: {10: 7},
    10: {},
}

# paper 94-103 data

dag = {
    1: {2: 2, 3: 2, 4: 2},
    2: {5: 3, 6: 1},
    3: {6: 1},
    4: {7: 2, 8: 2},
    5: {9: 2},
    6: {9: 2, 10: 2},
    7: {10: 2},
    8: {10: 3},
    9: {11: 2},
    10: {11: 4},
    11: {},
}
"""
# print(dag[1].keys())

# task(1-10) P1 P2 P3

"""运行时间"""
"""computation costs"""
"""
computation_costs = [
    [14, 16, 9],
    [13, 19, 18],
    [11, 13, 19],
    [13, 8, 17],
    [12, 13, 10],
    [13, 16, 9],
    [7, 15, 11],
    [5, 11, 14],
    [18, 12, 20],
    [21, 7, 16],
]


computation_costs = [
    [22, 21, 36],
    [22, 18, 18],
    [32, 27, 43],
    [7, 10, 4],
    [29, 27, 35],
    [26, 17, 24],
    [14, 25, 30],
    [29, 23, 36],
    [15, 21, 8],
    [13, 16, 33],
]

"""
"""
computation_costs = [
    [11, 9, 8],
    [5, 4, 4],
    [8, 7, 8],
    [9, 12, 12],
    [6, 6, 7],
    [5, 5, 5],
    [6, 7, 6],
    [8, 9, 9],
    [2, 3, 3],
    [8, 11, 9],
    [5, 5, 6],
    [9, 9, 9],
    [2, 3, 2],
    [1, 1, 1],
    [10, 9, 12],
    [6, 6, 7],
    [11, 8, 9],
    [8, 9, 7],
    [4, 5, 4],
    [4, 4, 4],
]
"""
"""
computation_costs = [
    [4, 4, 4],
    [5, 5, 5],
    [4, 6, 4],
    [3, 3, 3],
    [3, 5, 3],
    [3, 7, 2],
    [5, 8, 5],
    [2, 4, 5],
    [5, 6, 7],
    [3, 7, 5],
    [5, 6, 7]
]
"""


# The tasks of running on different procedures.
"""example p1=[{job:1,start:0,end:9},{}]"""


def avg_cost():
    """Computing the average computation costs of every task"""
    for n in range(len(computation_costs)):
        cost = round(sum(computation_costs[n]), 2)  # Keep two decimal places
        avg_costs.append(cost)
    return avg_costs


avg_cost()      # 调用执行


def rank__u(n):
    """Computing the priorities
    # Reverse topology sort is：10，9，8，7，6，5，4，3，2，1
    # Calculate in order rank_u(n_i) = (w_i ) ̅ + max(n_j∈succ(n_i)⁡{c(i,j) ) ̅ + rank_u (n_j )}"""
    # n = 10
    avg_cost()  # Computing the average computation costs of every task
    while n > 0:
        # print(dag[i])
        if len(dag[n]) == 0:  # no successors
            rank_u.append([n, avg_costs[n - 1]])
        else:  # have successors
            # Finding subsequent nodes j
            max_nj = 0
            for j in dag[n].keys():
                cij = dag[n][j] * 3  # communication cost of edge <i,j>
                # Finding subsequent nodes rank_uj.
                for k in range(len(rank_u)):
                    if j == rank_u[k][0]:
                        rank_uj = rank_u[k][1]
                        break
                if max_nj < cij + rank_uj:  # Take the maximum
                    max_nj = cij + rank_uj
            rank_u.append([n, avg_costs[n - 1] + max_nj])
        n -= 1
    return rank_u


rank__u(v)
# Sort by rank_u descending.
rank_u.sort(key=operator.itemgetter(1), reverse=True)
# rank_u_copy = rank_u.copy()
rank_u_copy = copy.deepcopy(rank_u)     # 2018.04.07


def pred_list():
    """Finding the Predecessor Node"""
    # rank__u(v)
    for m in range(len(rank_u)):
        job_ = rank_u[m][0]
        temp = []
        for j in range(len(dag)):
            if job_ in dag[j + 1].keys():
                sub_pred = j + 1
                temp.append(sub_pred)
        pred.append([job_, temp])
    return pred


pred_list()


def add_pi(pi_, job_, est_, eft_):
    """Join the task to the list and add the task to the schedule list"""
    if pi_ == 1:
        p1.append({'job': job_, 'start': est_, 'end': eft_})
        scheduler.append({job: 1})
    elif pi_ == 2:
        p2.append({'job': job_, 'start': est_, 'end': eft_})
        scheduler.append({job: 2})
    elif pi_ == 3:
        p3.append({'job': job_, 'start': est_, 'end': eft_})
        scheduler.append({job: 3})
    elif pi_ == 4:
        p4.append({'job': job_, 'start': est_, 'end': eft_})
        scheduler.append({job: 4})
    elif pi_ == 5:
        p5.append({'job': job_, 'start': est_, 'end': eft_})
        scheduler.append({job: 5})


def pred_max_nm(job_pred_):
    """The maximum time spent on a precursor node."""
    max_nm_ = 0
    for j in range(len(job_pred_)):
        # print(job_pred_[j])
        # Finding the completion time of the predecessor.1）Finding which processor the predecessor is on
        #  2）Finding the processor index location 3）Output the value of 'end'
        pred_pi = 0
        for k in range(len(rank_u_copy)):  # rank_u_copy
            if job_pred_[j] == rank_u_copy[k][0]:
                pred_pi = scheduler[k][job_pred_[j]]
                aft = 0
                if pred_pi == 1:
                    for m in range(len(p1)):
                        if p1[m]['job'] == job_pred_[j]:
                            aft = p1[m]['end']
                elif pred_pi == 2:
                    for m in range(len(p2)):
                        if p2[m]['job'] == job_pred_[j]:
                            aft = p2[m]['end']
                else:
                    for m in range(len(p3)):
                        if p3[m]['job'] == job_pred_[j]:
                            aft = p3[m]['end']
        # print(aft)
        # computing cmi
        if pi == pred_pi:
            cmi = 0
        else:
            cmi = dag[job_pred[j]][job]
        # print(cmi)

        if max_nm_ < aft + cmi:
            max_nm_ = aft + cmi
            # print(max_nm)
    return max_nm_


# Calculate the earliest start time  EST(n_i,p_j ) = max{avail[j], max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)}
# Calculate the earliest finish time  EFT(n_i,p_j)=w_(i,j) + EST(n_i,p_j)
while len(rank_u) > 0:
    """Select the first task schedule in the list each time"""
    job = rank_u.pop(0)[0]      # task id

    # if len(rank_u) == 9:  # The first task
    if len(rank_u) == v - 1:    # The first task
        est = 0
        # Find the job's minimum spend processor
        min_cost = computation_costs[job - 1][0]
        pi = 1  # default on p1 2018.04.06 add
        for i in range(3):
            if min_cost > computation_costs[job - 1][i]:
                min_cost = computation_costs[job - 1][i]
                pi = i+1      # Recorder the processor number
        eft = computation_costs[job-1][pi-1]        # computation_costs[job-1][pi-1]=wij
        # print([est, eft])
        add_pi(pi, job, est, eft)

    else:   # other tasks
        """First computing max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)}"""
        eft = M
        label = 0
        avail_pi = 0
        for pi in range(1, 4):  # Scheduling on different processors.
            est = 0
            job_pred = []
            for i in range(len(pred)):
                if job == pred[i][0]:  # Find the index position of predecessor i
                    job_pred = pred[i][1]
                    # print(job_pred)
                    pred_max_nm(job_pred)
            # Computing the earliest time that processor can handle of task job.
            if pi == 1 and len(p1) > 0:
                avail_pi = p1[-1]['end']
            elif pi == 2 and len(p2) > 0:
                avail_pi = p2[-1]['end']
            elif pi == 3 and len(p3) > 0:
                avail_pi = p3[-1]['end']

            max_nm = pred_max_nm(job_pred)
            if est < max(avail_pi, max_nm):
                est = max(avail_pi, max_nm)

            if eft > est + computation_costs[job-1][pi-1]:
                eft = est + computation_costs[job-1][pi-1]
                label = pi
        # update est
        est = eft - computation_costs[job-1][label-1]
        # print([est, eft])

        # Join the pi list
        add_pi(label, job, est, eft)

make_span = eft

print("-----------------------HEFT-----------------------")
print('scheduler =', scheduler)
print('p1 =', p1)
print('p2 =', p2)
print('p3 =', p3)
print('make_span =', make_span)
