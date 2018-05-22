# coding=utf-8
import operator
import heft_new
import time
import copy


class Cpop:

    def __init__(self, q, n, v):
        """"""
        self.heft = heft_new.Heft(q, n, v)
        self.pred = self.heft.pred_list()
        self.computation_costs = self.heft.computation_costs
        self.dag = self.heft.dag
        self.q = q
        self.n = n
        self.v = v
        self.Pi = {}
        self.rank_d = []
        self.scheduler = []  # Record the processor number where each task schedule is located
        self.priority = []
        self.cp = 0   # the priority of entry task
        self.set_cp = [1, ]     # A set of nodes with the same priority as the entry node
        self.rank_u_copy = self.heft.rank_u_copy
        self.label_pi = 0
        self.cp_min_costs = 0
        self.start_time = 0
        self.end_time = 0
        self.running_time = 0
        self.slr = 0
        self.speedup = 0

    def pred_rank_d(self, j, k, rank_d_copy):
        """Looking for the predecessor's rank_d"""
        rank_nj = 0  # Recoding the predecessor's rank_d
        for t in range(len(rank_d_copy)):  # m is the index of rank_d
            if self.heft.pred[j][1][k] == rank_d_copy[t][0]:
                rank_nj = rank_d_copy[t][1]
                break
        return rank_nj

    def get_max_pred_rank_d(self, j, n, rank_d_copy):
        """Get the maximum of predecessor's rank_d(n_j)+(w_j)̅+ c(j,i)̅ """
        max_nj = 0
        heft = self.heft
        for k in range(len(heft.pred[j][1])):  # k is the index of predecessor task list

            # find the predecessor's rank_d
            rank_nj = self.pred_rank_d(j, k, rank_d_copy)
            # find the predecessor's avg_cost
            w_j = heft.avg_costs[heft.pred[j][1][k] - 1]
            # find cji which from the predecessor to self node
            cji = self.dag[heft.pred[j][1][k]][n] * 3

            if max_nj < rank_nj + w_j + cji:
                max_nj = rank_nj + w_j + cji
        return max_nj

    def get_rank_d(self):
        """Computing Downward rank priorities, rank_d(n_i) = max(n_j∈pred(n_i)⁡{rank_d(n_j)+(w_j)̅+ c(j,i)̅ """
        n = 1  # task id
        v_ = self.v
        while n <= v_:
            rank_d_copy = copy.deepcopy(self.rank_d)  # 2018.04.11
            for j in range(len(self.pred)):  # j is the index of pred : 0 - 10
                if n == self.heft.pred[j][0]:  # find the predecessor's info of task i
                    # no predecessors
                    if len(self.heft.pred[j][1]) == 0:
                        self.rank_d.append([n, 0])
                    # have predecessors
                    else:
                        # get the maximum of predecessors ---rank_d(n_j)+(w_j)̅+ c(j,i)̅
                        max_nj = self.get_max_pred_rank_d(j, n, rank_d_copy)
                        self.rank_d.append([n, max_nj])
            n += 1
        return self.rank_d

    def get_compute_priority(self):
        """Computing the priorities  priority = rank_d + rank_u"""
        self.get_rank_d()

        for i in range(len(self.heft.rank_u_copy)):
            j = self.heft.rank_u_copy[i][0]  # the task id in rank_u
            self.priority.append([j, self.rank_d[j - 1][1] + self.heft.rank_u_copy[i][1]])
        self.priority.sort(key=operator.itemgetter(0))  # In ascending order of nodes
        self.cp = self.priority[0][1]
        return self.priority

    def get_set_cp(self):
        """get the set of SETcp"""
        self.get_compute_priority()
        n_k = 1  # task id
        v_ = self.v
        while n_k < v_:
            # find the successor of n_k
            for m in self.dag[n_k].keys():  # m is the successor of n_k
                n_j = m
                if self.priority[n_j - 1][1] == self.cp and n_j not in self.set_cp:  # Prevent duplicate additions
                    self.set_cp.append(n_j)
            n_k += 1
        return self.set_cp

    def critical_processor(self):
        """Select the optimal processor to perform the critical path task (in set_cp)"""
        self.get_set_cp()
        min_cost = self.heft.M
        pi = 0
        for i in range(len(self.computation_costs[0])):
            sum_cost = 0
            for j in range(len(self.set_cp)):
                job = self.set_cp[j]  # task id
                sum_cost += self.computation_costs[job - 1][i]
            if min_cost > sum_cost:
                min_cost = sum_cost
                pi = i + 1  # Record minimum spend processor number
        self.label_pi = pi
        self.cp_min_costs = min_cost
        return pi, min_cost

    def add_pi(self, pi_, job_, est_, eft_):
        """Join the task to the list and add the task to the schedule list"""
        list_pi = []
        if pi_ in self.Pi.keys():
            list_pi = list(self.Pi[pi_])
            list_pi.append({'job': job_, 'est': est_, 'end': eft_})
            self.Pi[pi_] = list_pi
            self.scheduler.append([job_, pi_])
        else:
            list_pi.append({'job': job_, 'est': est_, 'end': eft_})
            self.Pi[pi_] = list_pi
            self.scheduler.append([job_, pi_])

    def get_aft(self, job_pred_j):
        """"""
        aft = 0
        pred_pi = 0
        for k in range(len(self.rank_u_copy)):  # rank_u_copy
            if job_pred_j == self.rank_u_copy[k][0]:
                pred_pi = self.scheduler[k][job_pred_j]
                aft = 0
                for m in range(len(self.Pi[pred_pi])):
                    if self.Pi[pred_pi][m]['job'] == job_pred_j:
                        aft = self.Pi[pred_pi][m]['end']
        return aft, pred_pi

    def get_pred_aft(self, pred_pi, job_pred_j):
        """get the predecessor's aft"""
        aft = 0
        for m in range(len(self.Pi[pred_pi])):
            if self.Pi[pred_pi][m]['job'] == job_pred_j:
                aft = self.Pi[pred_pi][m]['end']
        return aft

    def compute_cmi(self, pi, pred_pi, job_pred_j, job):
        """computing the communication costs"""
        if pi == pred_pi:
            cmi = 0
        else:
            cmi = self.dag[job_pred_j][job]
        return cmi

    def pred_max_nm(self, pi, job, job_pred):
        """get the maximum costs of the predecessors，job_pred为job is a list of the predecessors"""
        max_nm = 0
        pred_pi = 0
        aft = 0
        for j in range(len(job_pred)):
            # Finding the completion time of the predecessor.1）Finding which processor the predecessor is on
            #  2）Finding the processor index location 3）Output the value of 'end'
            job_pred_j = []  # Record the list of predecessor task numbers to facilitate function call arguments
            for k in range(len(self.scheduler)):
                if job_pred[j] == self.scheduler[k][0]:
                    pred_pi = self.scheduler[k][1]
                    # get the predecessors's aft
                    job_pred_j = job_pred[j]
                    aft = self.get_pred_aft(pred_pi, job_pred_j)
            # computing cmi
            cmi = self.compute_cmi(pi, pred_pi, job_pred_j, job)

            # get the maximum of max_nm

            if max_nm < aft + cmi:
                max_nm = aft + cmi
        return max_nm

    def scheduling_critical_task(self, job):
        """Schedule critical path tasks"""
        label_pi = self.label_pi
        # Calculate the earliest time that the critical processor can handle
        avail_pi = 0
        if label_pi in self.Pi.keys():
            avail_pi = self.Pi[label_pi][-1]['end']

        # The maximum time spent on the predecessors task node
        max_nm = 0
        for ii in range(len(self.pred)):
            if job == self.pred[ii][0]:  # find the index ii of predecessors
                job_pred = self.pred[ii][1]  # list of the predecessors
                max_nm = self.pred_max_nm(label_pi, job, job_pred)

        est = max(avail_pi, max_nm)
        eft = est + self.computation_costs[job - 1][label_pi - 1]

        # added in p2 list
        self.add_pi(label_pi, job, est, eft)
        return eft

    def get_job_pred(self, job):
        """get the predecessor's list of job"""
        job_pred = []
        for i in range(len(self.pred)):
            if job == self.pred[i][0]:  # find the index i of the predecessor
                job_pred = self.pred[i][1]
        return job_pred

    def scheduling_uncritical_task(self, job):
        """scheduling uncritical task"""
        eft = self.heft.M
        label_pi = 0
        # the number of processor
        processor_num = len(self.computation_costs[0])

        # scheduling on different processors
        for pi in range(1, processor_num + 1):
            est = 0
            # get the predecessor's list of job
            job_pred = self.get_job_pred(job)

            # The earliest time that the computing processor can handle the task job
            avail_pi = 0
            if pi in self.Pi.keys():
                avail_pi = self.Pi[pi][-1]['end']
            # avail_pi = avail_est(pi)

            # computing max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)
            max_nm = self.pred_max_nm(pi, job, job_pred)

            # get eft and record the processor's id :label_pi
            if est < max(avail_pi, max_nm):
                est = max(avail_pi, max_nm)

            if eft > est + self.computation_costs[job - 1][pi - 1]:
                eft = est + self.computation_costs[job - 1][pi - 1]
                label_pi = pi
        # update est
        est = eft - self.computation_costs[job - 1][label_pi - 1]

        # add in pi list
        self.add_pi(label_pi, job, est, eft)
        return eft

    def get_unscheduled_job(self):
        """get_unscheduled_job"""
        un_schedule = []
        for k in range(len(self.priority)):
            un_schedule.append(self.priority[k][0])
        return un_schedule

    def judge_pred_complete_scheduling(self, length, job_temp, un_schedule):
        """judge pred complete scheduling"""
        # 1)find the pred task id
        num = 0  # record the complete scheduling number of the pred
        for j in range(length):
            # 2）judge pred complete scheduling of job_temp
            pred = self.pred[job_temp - 1][1][j]  # note! ! ! Three layers
            if pred not in un_schedule:
                num += 1
        return num

    def candidate_job(self, priority_copy, un_schedule):
        """unentry task，candidate_job"""
        job = 0
        for i in range(len(priority_copy)):
            job_temp = self.priority[i][0]  # candidate job id
            # judge pred complete scheduling
            length = len(self.pred[job_temp - 1][1])
            num = self.judge_pred_complete_scheduling(length, job_temp, un_schedule)
            if num == length:
                job = job_temp
                self.priority.pop(i)
                break
        return job

    def scheduling_unentry_task(self):
        """scheduling_unentry_task"""
        priority_copy = copy.deepcopy(self.priority)  # 2018.04.11

        # Predecessors sort in ascending order by task number
        self.pred.sort(key=operator.itemgetter(0))

        # get_unscheduled_job
        un_schedule = self.get_unscheduled_job()

        # candidate_job
        job = self.candidate_job(priority_copy, un_schedule)

        # critical_task
        if job in self.set_cp:
            eft = self.scheduling_critical_task(job)
        # uncritical_task
        else:
            eft = self.scheduling_uncritical_task(job)
        return eft

    def scheduling_entry_task(self):
        """scheduling_entry_task"""
        job = self.priority.pop(0)[0]
        est = 0
        eft = self.computation_costs[0][self.label_pi - 1]
        self.add_pi(self.label_pi, job, est, eft)
        return eft

    def cpop(self):
        """Execution algorithm"""
        # Initializes the priority queue, sorted by descending priority
        self.start_time = time.time()
        self.critical_processor()
        self.priority.sort(key=operator.itemgetter(1), reverse=True)
        eft = 0
        # while len(self.priority) > 0:  # have unscheduling tasks
        while len(self.priority) > 0:  # have unscheduling tasks
            # scheduling_entry_task
            if len(self.priority) == len(self.dag):
                eft = self.scheduling_entry_task()
            # scheduling_unentry_task
            else:
                eft = self.scheduling_unentry_task()
        self.end_time = time.time()
        self.running_time = int(round((self.end_time - self.start_time), 3) * 1000)
        self.slr = round(eft / self.cp_min_costs, 4)
        return eft


if __name__ == "__main__":
    q = 3
    n = 1
    v = 10
    cpop = Cpop(q, n, v)
    makespan = cpop.cpop()
    cp_min_costs = cpop.cp_min_costs
    print("-----------------------CPOP-----------------------")
    print('makespan =', makespan)
    print("cp_min_costs =", cp_min_costs)
    print("Running_time =", cpop.running_time)
    print("SLR =", cpop.slr)
    print("-----------------------CPOP-----------------------")





