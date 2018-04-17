# coding=utf-8
import os
dag = []
filename_ = 'dag_test.txt'
os.remove(filename_)  # 删除文件
for i in range(len(dag)):
    task_id = dag[i][0]
    for key, value in dag[i][1].items():
        succ_id = key
        succ_weight = value
        filename_ = 'dag_test.txt'   # 再新建文件
        with open(filename_, 'a') as file_object:
            info = str(task_id) + " " + str(succ_id) + " " + str(succ_weight) + "\n"
            file_object.write(info)

# 读DAG文件,构造dag
new_dag = {}
filename = 'dag_test.txt'
with open(filename, 'r') as file_object:
    lines = file_object.readlines()
    task_id_ = 0
    for i in range(len(dag) - 1):
        succ_dict = {}  # 后继字典
        for line in lines:
            line_list = line.split()  # 默认以空格为分隔符对字符串进行切片
            task_id = int(line_list[0])
            succ_id = int(line_list[1])
            succ_weight = int(line_list[2])

            if task_id == int(dag[i][0]):
                task_id_ = task_id
                succ_dict[succ_id] = succ_weight
        if task_id_ == int(dag[i][0]):
            new_dag[task_id_] = succ_dict
last_task_id = dag[-1][0]
new_dag[last_task_id] = {}
print(new_dag)
