"""The HEFT(Heterogeneous Earliest Finish Time) Scheduling Algorithm of DAGs"""
import operator
import copy


class Heft:

    def __init__(self, q, n, v):
        """"""
        self.Q = q
        self.n = n  # schedule which dag
        self.v = v  # the number of nodes
        self.dag = {}
        self.computation_costs = []
        self.M = 10000
        self.Pi = {}

        self.eft = 0  # Recording the make_span
        self.scheduler = []  # Recording the scheduling processors'id of different tasks.
        self.avg_costs = []  # Computing the average computation costs of every task.
        self.rank_u = []  # Recording the priorities
        self.pred = []  # Predecessor node list
        self.rank_u_copy = []

    def read_dag(self):
        """q is the number of processors, n is which graph"""
        # dag_ = {}
        filename = 'save_dag' + '\\' + 'v=' + str(self.v) + 'q=' + str(self.Q) + '\_' + str(self.n) + '_dag_q=' \
                   + str(self.Q) + '.txt'
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
                # dag_[task_id] = succ_dict
                self.dag[task_id] = succ_dict
            # dag_[task_id + 1] = {}
            self.dag[task_id + 1] = {}
        self.v = len(self.dag)
        return self.dag

    def read_computation_costs(self):
        """q is the number of processors, n is which graph"""
        # computation_costs_ = []
        filename = 'save_dag' + '\\' + 'v=' + str(self.v) + 'q=' + str(self.Q) + '\_' + str(self.n) \
                   + '_computation_costs_q=' + str(self.Q) + '.txt'
        with open(filename, 'r') as file_object:
            lines = file_object.readlines()
            for line in lines:
                line_list = line.split()
                temp_list = []
                for i in range(len(line_list)):
                    temp_list.append(int(line_list[i]))
                # computation_costs_.append(temp_list)
                self.computation_costs.append(temp_list)
        return self.computation_costs

    def get_dag_costs(self):
        self.read_dag()
        self.read_computation_costs()

    def get_v_q(self):
        # v = len(self.read_dag())
        v = self.v
        q = self.Q
        return v, q

    def avg_cost(self):
        """Computing the average computation costs of every task"""
        self.get_dag_costs()
        v = self.v
        computation_costs = self.computation_costs
        # v = self.get_v_q()[0]
        # computation_costs = self.dag_select()[1]
        # computation_costs = self.read_computation_costs()

        for t in range(v):
            cost = round(sum(computation_costs[t]), 2)  # Keep two decimal places
            self.avg_costs.append(cost)
        return self.avg_costs

    def rank__u(self):
        """Computing the priorities

         Calculate in order rank_u(n_i) = (w_i ) ̅ + max(n_j∈succ(n_i)⁡{c(i,j) ) ̅ + rank_u (n_j )}"""

        """Computing the average computation costs of every task"""
        self.avg_cost()
        dag = self.dag
        v = self.v
        # dag = self.dag_select(n)[0]
        # dag = self.read_dag()
        # v = self.get_v_q()[0]

        while v > 0:
            # print(dag[i])
            if len(dag[v]) == 0:  # no successors
                self.rank_u.append([v, self.avg_costs[v - 1]])
            else:  # have successors
                # Finding subsequent nodes j
                max_nj = 0
                for j in dag[v].keys():
                    cij = dag[v][j] * 3  # communication cost of edge <i,j>
                    # Finding subsequent nodes rank_uj.
                    for k in range(len(self.rank_u)):
                        if j == self.rank_u[k][0]:
                            rank_uj = self.rank_u[k][1]
                            break
                    if max_nj < cij + rank_uj:  # Take the maximum
                        max_nj = cij + rank_uj
                self.rank_u.append([v, self.avg_costs[v - 1] + max_nj])
            v -= 1
        return self.rank_u

    def execute_rank__u(self):
        self.rank__u()
        # Sort by rank_u descending.
        self.rank_u.sort(key=operator.itemgetter(1), reverse=True)
        # rank_u_copy = rank_u.copy()
        self.rank_u_copy = copy.deepcopy(self.rank_u)  # 2018.04.07
        return self.rank_u_copy

    def pred_list(self):
        """Finding the Predecessor Node"""
        # dag = self.dag_select(n)[0]
        # dag = self.read_dag()
        dag = self.dag
        self.execute_rank__u()

        for m in range(len(self.rank_u)):
            job_ = self.rank_u[m][0]
            temp = []
            for j in range(len(dag)):
                if job_ in dag[j + 1].keys():
                    sub_pred = j + 1
                    temp.append(sub_pred)
            self.pred.append([job_, temp])
        
        return self.pred

    def add_pi(self, pi_, job_, est_, eft_):
        """Join the task to the list and add the task to the schedule list"""
        list_pi = []
        if pi_ in self.Pi.keys():
            list_pi = list(self.Pi[pi_])
            list_pi.append({'job': job_, 'est': est_, 'end': eft_})
            self.Pi[pi_] = list_pi
            self.scheduler.append({job_: pi_})
        else:
            list_pi.append({'job': job_, 'est': est_, 'end': eft_})
            self.Pi[pi_] = list_pi
            self.scheduler.append({job_: pi_})

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

    def pred_max_nm(self, pi_, job_pred_, job_):
        """The maximum time spent on a precursor node."""
        # dag = self.dag_select(n)[0]
        # dag = self.read_dag()
        dag = self.dag

        max_nm_ = 0
        for j in range(len(job_pred_)):
            # print(job_pred_[j])
            # Finding the completion time of the predecessor.1）Finding which processor the predecessor is on
            #  2）Finding the processor index location 3）Output the value of 'end'
            job_pred_j = job_pred_[j]
            """get aft"""
            aft, pred_pi = self.get_aft(job_pred_j)

            # computing cmi
            if pi_ == pred_pi:
                cmi = 0
            else:
                cmi = dag[job_pred_[j]][job_]
            if max_nm_ < aft + cmi:
                max_nm_ = aft + cmi
        return max_nm_

    def get_min_comp_costs(self):
        """Minimizes the cumulative of the computation costs"""
        # computation_costs = self.dag_select(n)[1]
        # computation_costs = self.read_computation_costs()
        computation_costs = self.computation_costs

        min_comp_costs = 100000
        for p in range(len(computation_costs[0])):
            sum_cost = 0
            for j in range(len(computation_costs)):
                sum_cost += computation_costs[j][p]
            if min_comp_costs > sum_cost:
                min_comp_costs = sum_cost
        return min_comp_costs

    def heft(self):
        """"""
        # Calculate the earliest start time  EST(n_i,p_j ) = max{avail[j], max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)}
        # Calculate the earliest finish time  EFT(n_i,p_j)=w_(i,j) + EST(n_i,p_j)

        self.pred_list()
        # computation_costs = self.dag_select(n)[1]
        # computation_costs = self.read_computation_costs()
        computation_costs = self.computation_costs

        # v = self.get_v_q()[0]
        v = self.v

        eft = 0
        while len(self.rank_u) > 0:
            """Select the first task schedule in the list each time"""
            job = self.rank_u.pop(0)[0]  # task id
            # pred_list()

            if len(self.rank_u) == v - 1:  # The first task
                est = 0
                # Find the job's minimum spend processor
                min_cost = computation_costs[job - 1][0]
                pi = 1  # default on p1 2018.04.06 add
                for i in range(len(computation_costs[0])):
                    if min_cost > computation_costs[job - 1][i]:
                        min_cost = computation_costs[job - 1][i]
                        pi = i + 1  # Recorder the processor number
                eft = computation_costs[job - 1][pi - 1]  # computation_costs[job-1][pi-1]=wij
                # print([est, eft])
                self.add_pi(pi, job, est, eft)

            else:  # other tasks
                """First computing max(n_m∈pred(n_i)){AFT(n_m ) + c_(m,i)}"""
                eft = self.M
                label = 0
                avail_pi = 0
                for pi in range(1, self.Q + 1):  # Scheduling on different processors.
                    est = 0
                    job_pred = []
                    max_nm = 0
                    for i in range(len(self.pred)):
                        if job == self.pred[i][0]:  # Find the index position of predecessor i
                            job_pred = self.pred[i][1]
                            max_nm = self.pred_max_nm(pi, job_pred, job)

                    # Computing the earliest time that processor can handle of task job.
                    avail_pi = 0
                    if pi in self.Pi.keys():
                        avail_pi = self.Pi[pi][-1]['end']

                    # max_nm = pred_max_nm(pi, job_pred, job)
                    if est < max(avail_pi, max_nm):
                        est = max(avail_pi, max_nm)

                    if eft > est + computation_costs[job - 1][pi - 1]:
                        eft = est + computation_costs[job - 1][pi - 1]
                        label = pi
                # update est
                est = eft - computation_costs[job - 1][label - 1]
                # print([est, eft])

                # Join the pi list
                self.add_pi(label, job, est, eft)
        return eft


if __name__ == "__main__":
    Q = 4
    n = 1
    V = 20
    heft = Heft(Q, n, V)
    make_span = heft.heft()

    print("-----------------------HEFT-----------------------")

    print('make_span =', make_span)

    print("-----------------------HEFT-----------------------")


