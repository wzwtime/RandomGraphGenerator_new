# coding=utf-8
"""CPOP(Critical Path On a Processor)2018-3-15"""
import operator
import heft
import time
import copy

start_cpop = time.time()
# import matplotlib as plt

# Computing rank_d
rank_d = []
v = heft.v
p1 = []
p2 = []
p3 = []
scheduler = []      # Record the processor number where each task schedule is located
priority = []


def pred_rank_d(j, k, rank_d_copy):
    """Looking for the predecessor's rank_d"""
    rank_nj = 0  # Recoding the predecessor's rank_d
    for t in range(len(rank_d_copy)):  # m is the index of rank_d
        if heft.pred[j][1][k] == rank_d_copy[t][0]:
            rank_nj = rank_d_copy[t][1]
            break
    return rank_nj


def get_max_pred_rank_d(j, n, rank_d_copy):
    """Get the maximum of predecessor's rank_d(n_j)+(w_j)̅+ c(j,i)̅ """
    max_nj = 0
    for k in range(len(heft.pred[j][1])):  # k is the index of predecessor task list

        # find the predecessor's rank_d
        rank_nj = pred_rank_d(j, k, rank_d_copy)
        # find the predecessor's avg_cost
        w_j = heft.avg_costs[heft.pred[j][1][k] - 1]
        # find cji which from the predecessor to self node
        cji = heft.dag[heft.pred[j][1][k]][n] * 3

        if max_nj < rank_nj + w_j + cji:
            max_nj = rank_nj + w_j + cji
    return max_nj


def get_rank_d(v_):
    """Computing Downward rank priorities, rank_d(n_i) = max(n_j∈pred(n_i)⁡{rank_d(n_j)+(w_j)̅+ c(j,i)̅ """
    n = 1   # task id
    while n <= v_:
        rank_d_copy = copy.deepcopy(rank_d)     # 2018.04.11
        for j in range(len(heft.pred_list())):  # j is the index of pred : 0 - 10
            if n == heft.pred[j][0]:    # find the predecessor's info of task i
                # no predecessors
                if len(heft.pred[j][1]) == 0:
                    rank_d.append([n, 0])
                # have predecessors
                else:
                    # get the maximum of predecessors ---rank_d(n_j)+(w_j)̅+ c(j,i)̅
                    max_nj = get_max_pred_rank_d(j, n, rank_d_copy)
                    rank_d.append([n, max_nj])
        n += 1
    return rank_d


get_rank_d(v)


def get_compute_priority():
    """Computing the priorities  priority = rank_d + rank_u"""
    for i in range(len(heft.rank_u_copy)):
        j = heft.rank_u_copy[i][0]  # the task id in rank_u
        priority.append([j, rank_d[j-1][1] + heft.rank_u_copy[i][1]])
    priority.sort(key=operator.itemgetter(0))   # In ascending order of nodes


get_compute_priority()

# complete the set of SETcp
cp = priority[0][1]     # the priority of entry task
set_cp = [1, ]     # A set of nodes with the same priority as the entry node


def get_set_cp(v_):
    """get the set of SETcp"""
    n_k = 1  # task id
    while n_k < v_:
        # find the successor of n_k
        for m in heft.dag[n_k].keys():  # m is the successor of n_k
            n_j = m
            if priority[n_j - 1][1] == cp and n_j not in set_cp:  # Prevent duplicate additions
                set_cp.append(n_j)
        n_k += 1


get_set_cp(v)
# print(set_cp)

# Select critical path processor to minimize total processing time


def critical_processor():
    """Select the optimal processor to perform the critical path task (in set_cp)"""
    min_cost = heft.M
    pi = 0
    for i in range(len(heft.computation_costs[0])):
        sum_cost = 0
        for j in range(len(set_cp)):
            job = set_cp[j]    # task id
            sum_cost += heft.computation_costs[job-1][i]
        if min_cost > sum_cost:
            min_cost = sum_cost
            pi = i + 1  # Record minimum spend processor number
    return pi, min_cost


min_cost_pi = critical_processor()[0]


def add_pi(pi, job, est, eft):
    """Join the task to the list and add the task to the schedule list"""

    if pi == 1:
        p1.append({'job': job, 'start': est, 'end': eft})
        scheduler.append([job, 1])
    elif pi == 2:
        p2.append({'job': job, 'start': est, 'end': eft})
        scheduler.append([job, 2])
    else:
        p3.append({'job': job, 'start': est, 'end': eft})
        scheduler.append([job, 3])


def get_pred_aft(pred_pi, job_pred_j):
    """get the predecessor's aft"""
    aft = 0
    if pred_pi == 1:
        for i in range(len(p1)):
            if p1[i]['job'] == job_pred_j:
                aft = p1[i]['end']
    elif pred_pi == 2:
        for i in range(len(p2)):
            if p2[i]['job'] == job_pred_j:
                aft = p2[i]['end']
    else:
        for i in range(len(p3)):
            if p3[i]['job'] == job_pred_j:
                aft = p3[i]['end']
    return aft


def compute_cmi(pi, pred_pi, job_pred_j, job):
    """computing the communication costs"""
    if pi == pred_pi:
        cmi = 0
    else:
        cmi = heft.dag[job_pred_j][job]
    return cmi


def pred_max_nm(pi, job, job_pred):
    """get the maximum costs of the predecessors，job_pred为job is a list of the predecessors"""
    max_nm = 0
    pred_pi = 0
    aft = 0
    for j in range(len(job_pred)):
        # Finding the completion time of the predecessor.1）Finding which processor the predecessor is on
        #  2）Finding the processor index location 3）Output the value of 'end'
        job_pred_j = []  # Record the list of predecessor task numbers to facilitate function call arguments
        for k in range(len(scheduler)):
            if job_pred[j] == scheduler[k][0]:
                pred_pi = scheduler[k][1]
                # get the predecessors's aft
                job_pred_j = job_pred[j]
                aft = get_pred_aft(pred_pi, job_pred_j)
        # computing cmi
        cmi = compute_cmi(pi, pred_pi, job_pred_j, job)

        # get the maximum of max_nm

        if max_nm < aft + cmi:
            max_nm = aft + cmi
    return max_nm


def scheduling_critical_task(job):
    """Schedule critical path tasks"""
    label_pi = min_cost_pi
    # Calculate the earliest time that the critical processor can handle
    avail_pi = avail_est(label_pi)
    # print(avail_pi)

    # The maximum time spent on the predecessors task node
    max_nm = 0
    for ii in range(len(heft.pred)):
        if job == heft.pred[ii][0]:  # find the index ii of predecessors
            job_pred = heft.pred[ii][1]  # list of the predecessors
            max_nm = pred_max_nm(label_pi, job, job_pred)

    est = max(avail_pi, max_nm)
    eft = est + heft.computation_costs[job - 1][label_pi - 1]

    # added in p2 list
    add_pi(label_pi, job, est, eft)
    return eft


def avail_est(pi):
    """The earliest time that the computing processor can handle the task job,pi is the id of processor"""
    avail_pi = 0
    if pi == 1 and len(p1) > 0:
        avail_pi = p1[-1]['end']
    elif pi == 2 and len(p2) > 0:
        avail_pi = p2[-1]['end']
    elif pi == 3 and len(p3) > 0:
        avail_pi = p3[-1]['end']
    return avail_pi


def get_job_pred(job):
    """get the predecessor's list of job"""
    job_pred = []
    for i in range(len(heft.pred)):
        if job == heft.pred[i][0]:  # find the index i of the predecessor
            job_pred = heft.pred[i][1]
    return job_pred


def scheduling_uncritical_task(job):
    """scheduling uncritical task"""
    eft = heft.M
    label_pi = 0
    # the number of processor
    processor_num = len(heft.computation_costs[0])

    # scheduling on different processors
    for pi in range(1, processor_num + 1):
        est = 0
        # get the predecessor's list of job
        job_pred = get_job_pred(job)

        # The earliest time that the computing processor can handle the task job
        avail_pi = avail_est(pi)

        # computing max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)
        max_nm = pred_max_nm(pi, job, job_pred)

        # get eft and record the processor's id :label_pi
        if est < max(avail_pi, max_nm):
            est = max(avail_pi, max_nm)

        if eft > est + heft.computation_costs[job - 1][pi - 1]:
            eft = est + heft.computation_costs[job - 1][pi - 1]
            label_pi = pi
    # update est
    est = eft - heft.computation_costs[job - 1][label_pi - 1]

    # add in pi list
    add_pi(label_pi, job, est, eft)
    return eft


def judge_pred_complete_scheduling(length, job_temp, un_schedule):
    """judge pred complete scheduling"""
    # 1)find the pred task id
    num = 0  # record the complete scheduling number of the pred
    for j in range(length):
        # 2）judge pred complete scheduling of job_temp
        pred = heft.pred[job_temp - 1][1][j]  # note! ! ! Three layers
        if pred not in un_schedule:
            num += 1
    return num


def candidate_job(priority_copy, un_schedule):
    """unentry task，candidate_job"""
    job = 0
    for i in range(len(priority_copy)):
        job_temp = priority[i][0]  # candidate job id
        # judge pred complete scheduling
        length = len(heft.pred[job_temp - 1][1])
        num = judge_pred_complete_scheduling(length, job_temp, un_schedule)
        if num == length:
            job = job_temp
            priority.pop(i)
            break
    return job


def get_unscheduled_job():
    """get_unscheduled_job"""
    un_schedule = []
    for k in range(len(priority)):
        un_schedule.append(priority[k][0])
    return un_schedule


def scheduling_unentry_task():
    """scheduling_unentry_task"""
    priority_copy = copy.deepcopy(priority)     # 2018.04.11

    # Predecessors sort in ascending order by task number
    heft.pred.sort(key=operator.itemgetter(0))

    # get_unscheduled_job
    un_schedule = get_unscheduled_job()

    # candidate_job
    job = candidate_job(priority_copy, un_schedule)

    # critical_task
    if job in set_cp:
        eft = scheduling_critical_task(job)
    # uncritical_task
    else:
        eft = scheduling_uncritical_task(job)
    return eft


def scheduling_entry_task():

    """scheduling_entry_task"""
    job = priority.pop(0)[0]
    est = 0
    eft = heft.computation_costs[0][min_cost_pi - 1]
    add_pi(min_cost_pi, job, est, eft)


def cpop():
    """Execution algorithm"""
    # Initializes the priority queue, sorted by descending priority
    priority.sort(key=operator.itemgetter(1), reverse=True)
    eft = 0

    while len(priority) > 0:   # have unscheduling tasks
        # scheduling_entry_task
        if len(priority) == len(heft.dag):
            scheduling_entry_task()
        # scheduling_unentry_task
        else:
            eft = scheduling_unentry_task()
    return eft


make_span = cpop()

end_cpop = time.time()
running_time_cpop = int(round((end_cpop - start_cpop), 3) * 1000)
print("time=", running_time_cpop)
print("-----------------------CPOP------------------------")
# print('scheduler:', scheduler)
# print('p1:', p1)
# print('p2:', p2)
# print('p3:', p3)
print('make_span:', make_span)

min_costs = critical_processor()[1]
slr_heft = round(heft.make_span / min_costs, 2)
slr_cpop = round(make_span / min_costs, 2)

min_comp_costs = heft.get_min_comp_costs()
speedup_heft = round(min_comp_costs / heft.make_span, 2)
speedup_cpop = round(min_comp_costs / make_span, 2)


def write_slr_speedup():
    """computing schedule length ratio"""
    filename = "slr_speedup_heft_cpop.txt"
    with open(filename, 'a') as file_object:
        info = str(v) + "  " + str(min_costs) + "  " + str(min_comp_costs) + "  " + str(heft.make_span) + "  " \
               + str(make_span) + "  " + str(slr_heft) + "  " + str(slr_cpop) + "  " + str(speedup_heft) + "  " \
               + str(speedup_cpop) + "\n"
        """+ "  " + str(heft.running_time_heft) + "  " + str(running_time_cpop)"""
        file_object.write(info)


write_slr_speedup()

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


# read_slr_speedup(SLR_heft, Speedup_heft)
# read_slr_speedup(SLR_cpop, Speedup_cpop)

# print("SLR_heft=", SLR_heft)
# print("SLR_heft=", SLR_cpop)
# print("speedup_heft=", Speedup_heft)
# print("speedup_heft=", Speedup_cpop)


heft_info = []
cpop_info = []


def slr_info(slr_list, info):
    """computing avg slr"""
    for key, values in slr_list.items():
        print(key, values)
        print(len(values))
        temp_info = []
        min_info = min(values)
        temp_info.append(min_info)
        max_info = max(values)
        temp_info.append(max_info)
        avg_slr_ = round(sum(values) / len(values), 2)
        temp_info.append(avg_slr_)
        info.append(temp_info)


# slr_info(SLR_heft, heft_info)
# print("heft =", heft_info)
#
# slr_info(SLR_cpop, cpop_info)
# print("cpop =", cpop_info)


def write_ccr(ccr):
    """write ccr in file"""
    filename = "ccr.txt"
    with open(filename, 'a') as file_object:
        info = str(ccr) + "  " + str(v) + "  " + str(min_costs) + "  " + str(heft.make_span) + "  " + str(make_span) \
               + "  " + str(slr_heft) + "  " + str(slr_cpop) + "\n"
        """+ "  " + str(heft.running_time_heft) + "  " + str(running_time_cpop)"""
        file_object.write(info)


# write_ccr(10.0)


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


def write_slr_beta(beta):
    """write slr_beta in file"""
    filename = "slr_beta.txt"
    with open(filename, 'a') as file_object:
        info = str(beta) + "  " + str(v) + "  " + str(min_costs) + "  " + str(heft.make_span) + "  " + str(make_span) \
               + "  " + str(slr_heft) + "  " + str(slr_cpop) + "\n"
        file_object.write(info)


write_slr_beta(1.0)
