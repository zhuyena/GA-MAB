import time
from math import inf
import numpy as np
from Myproblem import MyProblem

# 实例文件路径
from GA import GA
Node_distribution = 'singlecenter'
# TSP_D_Istance = ['uniform-71-n50','uniform-72-n20','uniform-73-n50','uniform-74-n50','uniform-75-n50','uniform-76-n50','uniform-77-n50','uniform-78-n50','uniform-79-n50','uniform-80-n50']
TSP_D_Istance = ['singlecenter-61-n20']
with open('../output_results/experiment_results.txt', 'w') as file:
    file.write(f'Node_distribution :{Node_distribution}\n\n')
for instance in TSP_D_Istance:
    file_path = 'TSP-D-Instances-master/' + Node_distribution + '/' + instance + '.txt'
    # 存储结果路径填
    out_path = '../output_results'
    problem = MyProblem(file_path)  # define problem
    popsize = problem.popsize  # 种群大小
    max_iter = problem.Iter  # 最大迭代次数
    total_cost = 0
    total_time = 0
    best_cost = inf
    best_solution = []
    repeat = 0  # 遗传算法运行次数
    # 打开一个文本文件，以写入模式 ("w") 打开
    with open('../output_results/experiment_results.txt', 'a') as file:
        file.write(f'Problem_name :{instance}\n')
        file.write(f'alpha :{problem.alpha}\n')
    while repeat < problem.run_times:
        # print('{}th run, benchmark:{}'.format(repeat + 1, problemName))
        print('{}th run, benchmark:{}'.format(repeat + 1, instance))
        myAlgorithm = GA(problem, popsize, max_iter)
        # 记录运行时间
        start_time = time.time()
        myAlgorithm.run()
        end_time = time.time()
        execution_time = end_time - start_time
        current_cost = myAlgorithm.low_cost
        current_solution = myAlgorithm.best_solution
        with open('../output_results/experiment_results.txt', 'a') as file:
            file.write(f'{repeat + 1}th run:\n')
            file.write(f'current cost:{current_cost}\n')
            file.write(f'current solution:{current_solution}\n')
            file.write(f'execution time:{execution_time}\n')
        # use_times = myAlgorithm.use_times
        # rewards = myAlgorithm.reward
        if current_cost <= best_cost:
            best_cost = current_cost
            best_solution = current_solution
        total_cost += current_cost
        total_time += execution_time
        # myAlgorithm.plot_map()
        myAlgorithm.plot_convergence()
        print('current cost is', current_cost)
        print('current solution is', best_solution)
        # print('operators rewards are', rewards)
        print('execution_time is ', execution_time)
        # print('operators use times are', use_times)
        print('fitness/iter = ', myAlgorithm.fitnessvalue)
        print('\n')
        repeat += 1
    # mean solution：进行多次实验的平均结果
    mean_cost = total_cost / problem.run_times
    mean_time = total_time / problem.run_times
    # print(problemName, 'finished')
    print('best cost is ', best_cost)
    print('mean cost is ', mean_cost)
    print('mean time is ', mean_time)
    with open('../output_results/experiment_results.txt', 'a') as file:
        file.write(f'best cost:{best_cost}\n')
        file.write(f'mean cost:{mean_cost}\n')
        file.write(f'mean time:{mean_time}\n\n')










