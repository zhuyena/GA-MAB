import copy
import time
import numpy as np
from matplotlib import pyplot as plt
from Individual import Individual
from population import Population


class GA:
    def __init__(self, problem, popsize, max_iter):
        self.assigned_Qt = None
        self.best_solution = None
        self.low_cost = None
        self.problem = problem
        self.popsize = popsize
        self.Pop = []  # 只有卡车路径的种群
        self.assign_pop = []  # 分配完的种群，包含卡车路径与无人机路径
        self.idv = Individual()  # 个体
        self.populations = Population(popsize, problem)
        self.max_iter = max_iter
        self.final_pop = []
        self.final_pop_value = []
        self.final_pop_solution = []
        self.operations = ['swap2', '3opt', 'relocate', 'swap', '2opt']
        self.reward = [0]*len(self.operations)
        self.prop = [1/len(self.operations)]*len(self.operations)  # 各个算子初始化的选择概率
        self.use_times = [0]*len(self.operations)  # 算子使用次数
        self.fitnessvalue = []
        self.epsilon = 0.1  # 探索概率
        self.decline = 0.9

    def run(self):
        self.Pop = self.populations.creat_pop()  # 生成一个都由卡车服务的种群
        start = time.time()
        # self.assign_pop = self.populations.ep_assigned(self.Pop, self.final_pop, self.final_pop_value, self.final_pop_solution)
        self.assign_pop = self.populations.assigned_noadjust(self.Pop, self.final_pop, self.final_pop_value, self.final_pop_solution)
        end = time.time()
        print('assigned time is ', end - start)
        current_pop = self.assign_pop
        gen = 0
        while gen < self.max_iter:
            gen += 1
            child_pop = self.populations.next_pop(current_pop)  # 交叉变异
            current_pop = self.UpdatePopulation(current_pop, child_pop)
            current_pop[0] = current_pop[0][:self.popsize]
            current_pop[1] = current_pop[1][:self.popsize]
            current_pop[2] = current_pop[2][:self.popsize]

            current_pop = self.local_search(current_pop, child_pop)  # 局部搜索（包含了精英选择以及局部搜索完后的种群更新操作）
            # current_pop = self.local_search_rls(current_pop, child_pop)

            self.fitnessvalue.append(current_pop[1][0])
        self.low_cost = current_pop[1][0]
        self.best_solution = current_pop[2][0]

    def local_search(self, current_pop, child_pop):
        # 一个100次、10个10次进行对比
        operations = self.operations
        K = len(operations)
        pop_to_ls = self.UpdatePopulation(current_pop, child_pop)  # 更新种群
        elite_ind = pop_to_ls[0][:10]
        elite_value = pop_to_ls[1][:10]
        elite_solution = pop_to_ls[2][:10]
        T = 15
        new_tour = []
        new_value = []
        new_solution = []
        eps = 0.1
        for idvi in range(5):  # 10个精英个体
            current_idv = elite_ind[idvi]  # 当前个体
            current_value = elite_value[idvi]
            Q = [0] * len(self.operations)  # 每个算子的平均奖励值
            count = [0] * len(self.operations)  # 算子使用次数，初始化为0
            # 对每个个体进行T次摇臂
            for t in range(T):
                if np.random.random() < eps:  # 探索
                    k = np.random.randint(0, K)  # 随机选择
                    # self.use_times[k] += 1
                else:
                    if all(x == 0 for x in Q):
                        # 所有元素均为 0，随机选择一个元素的下标
                        k = np.random.randint(0, K)
                        # self.use_times[k] += 1
                    else:
                        # 选择最大元素的下标
                        k = Q.index(max(Q))
                        # self.use_times[k] += 1
                operator = operations[k]  # 选择k对应的算子
                # idv_copy = copy.deepcopy(pop_to_ls[0])  ####
                # idv_value_copy = copy.deepcopy(pop_to_ls[1])  ####
                # # 局部搜索
                # origin_value = idv_value_copy[idvi]  # 在运用算子之前的value值  ####
                current_idv_change = self.problem.neighborhoods1(current_idv, operator)  # 第k个算子修改当前个体
                # 局部搜索后的个体解码
                [new_tour, new_value, new_solution] = self.populations.ep_assigned([current_idv_change], new_tour, new_value,new_solution)
                [new_tour, new_value, new_solution] = self.populations.assigned_noadjust([current_idv_change], new_tour,
                                                                                   new_value, new_solution)

                r = max(current_value-new_value[-1], 0)
                # 更新Q值
                Q[k] = (Q[k]*count[k]+r)/(count[k]+1)
                count[k] = count[k] + 1

            self.use_times = [x + y for x, y in zip(self.use_times, count)]

            #     if new_value[-1] < origin_value:  # 是否优化了结果
            #         self.reward[j] = self.decline * self.reward[j] + (origin_value - new_value[-1])  # 衰减后将优化的距离作为奖励值
            #         self.prop = [self.epsilon / len(self.operations)] * len(self.operations)  # 将其余的算子概率设置成ε/k
            #         self.prop[j] = 1 - self.epsilon + (self.epsilon / len(self.operations))  # 将当前算子概率设置为1-ε+ε/k
            #
            #     self.use_times[j] += 1
            #     break
            #
            # p = np.random.random()  # 随机生成一个概率
            # sum_prop = 0
            # for j in range(len(operations)):
            #     sum_prop += self.prop[j]
            #     if p <= sum_prop:
            #         operator = operations[j]  # 选择当前概率对应的算子
            #         idv_copy = copy.deepcopy(pop_to_ls[0])
            #         idv_value_copy = copy.deepcopy(pop_to_ls[1])
            #         # 局部搜索
            #         origin_value = idv_value_copy[idvi]  # 在运用算子之前的value值
            #         current_solution = self.problem.neighborhoods1(idv_copy, idvi, operator)  # 返回新的种群和优化的染色体数
            #         # 局部搜索后的个体解码
            #         [new_tour, new_value, new_solution] = self.populations.assigned(current_solution, new_tour, new_value, new_solution)
            #
            #         if new_value[-1] < origin_value:  # 是否优化了结果
            #             self.reward[j] = self.decline * self.reward[j] + (origin_value - new_value[-1])  # 衰减后将优化的距离作为奖励值
            #             self.prop = [self.epsilon/len(self.operations)] * len(self.operations)  # 将其余的算子概率设置成ε/k
            #             self.prop[j] = 1 - self.epsilon + (self.epsilon/len(self.operations))  # 将当前算子概率设置为1-ε+ε/k
            #
            #         self.use_times[j] += 1
            #         break
            #     else:
            #         pass
        new_pop = [new_tour, new_value, new_solution]
        final_pop = self.UpdatePopulation(pop_to_ls, new_pop)
        final_pop[0] = final_pop[0][:self.popsize]
        final_pop[1] = final_pop[1][:self.popsize]
        final_pop[2] = final_pop[2][:self.popsize]

        return final_pop

    def local_search_rls(self, current_pop, child_pop):
        # 一个100次、10个10次进行对比
        operations = self.operations
        K = len(operations)
        pop_to_ls = self.UpdatePopulation(current_pop, child_pop)  # 更新种群
        elite_ind = pop_to_ls[0][:10]
        elite_value = pop_to_ls[1][:10]
        elite_solution = pop_to_ls[2][:10]
        T = 20
        new_tour = []
        new_value = []
        new_solution = []
        eps = 0.1
        for idvi in range(5):  # 10个精英个体
            current_idv = elite_ind[idvi]  # 当前个体
            current_value = elite_value[idvi]
            # Q = [0] * len(self.operations)  # 每个算子的平均奖励值
            # count = [0] * len(self.operations)  # 算子使用次数，初始化为0
            # 对每个个体进行T次摇臂
            for t in range(T):
                # if np.random.random() < eps:  # 探索
                #     k = np.random.randint(0, K)  # 随机选择
                # else:
                #     if all(x == 0 for x in Q):
                #         # 所有元素均为 0，随机选择一个元素的下标
                #         k = np.random.randint(0, K)
                #     else:
                #         # 选择最大元素的下标
                #         k = Q.index(max(Q))
                k = np.random.randint(0, K)
                operator = operations[k]  # 选择k对应的算子
                # idv_copy = copy.deepcopy(pop_to_ls[0])  ####
                # idv_value_copy = copy.deepcopy(pop_to_ls[1])  ####
                # # 局部搜索
                # origin_value = idv_value_copy[idvi]  # 在运用算子之前的value值  ####
                current_idv_change = self.problem.neighborhoods1(current_idv, operator)  # 第k个算子修改当前个体
                # 局部搜索后的个体解码
                [new_tour, new_value, new_solution] = self.populations.assigned([current_idv_change], new_tour, new_value,
                                                                                new_solution)
                # r = max(current_value-new_value[-1], 0)
                # # 更新Q值
                # Q[k] = (Q[k]*count[k]+r)/(count[k]+1)
                # count[k] = count[k] + 1

        new_pop = [new_tour, new_value, new_solution]
        final_pop = self.UpdatePopulation(pop_to_ls, new_pop)
        final_pop[0] = final_pop[0][:self.popsize]
        final_pop[1] = final_pop[1][:self.popsize]
        final_pop[2] = final_pop[2][:self.popsize]

        return final_pop

    def plot_map(self):
        x_values = [self.problem.location[i][0] for i in range(len(self.problem.location))]
        x_values = [float(x) for x in x_values]
        y_values = [self.problem.location[i][1] for i in range(len(self.problem.location))]
        y_values = [float(x) for x in y_values]
        city_coordinates = zip(x_values, y_values)
        index = np.arange(self.problem.node_num)
        index.tolist()
        city = dict(zip(index,city_coordinates))
        #  绘制卡车路线
        X = []
        Y = []
        x = []
        y = []
        text_list = []
        text_list_total = []
        for a in range(len(city)):
            X.append(city[a][0])
            Y.append(city[a][1])
            text_list_total.append(str(a))

        for v in self.best_solution[0]:
            x.append(city[v][0])
            y.append(city[v][1])
            text_list.append(str(v))

        for i in range(len(text_list_total)):
            plt.text(X[i], Y[i], text_list_total[i], ha='center', va='center_baseline')

        plt.plot(x, y, 'c-', linewidth=2, markersize=12)
        #  无人机架次
        for s in self.best_solution[1]:
            x1 = []
            y1 = []
            for s1 in s:
                x1.append(city[s1][0])
                y1.append(city[s1][1])
            plt.plot(x1, y1, 'r--', linewidth=2, markersize=12)
        # 显示图形
        plt.show()

    def plot_convergence(self):
        # 模拟一些示例数据作为收敛曲线上的点
        num_generations = self.max_iter
        fitness_values = self.fitnessvalue
        # 绘制收敛曲线
        plt.figure()
        plt.plot(range(num_generations), fitness_values, marker='o', linestyle='-')
        plt.title('Genetic Algorithm Convergence Curve for TSP-D(no-ls,d-68-20)')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.grid(True)
        plt.show()

    def UpdatePopulation(self, current_pop, child_pop):
        Original_Pop = []
        pop_to_ls = []
        original_tour = current_pop[0] + child_pop[0]
        original_value = current_pop[1] + child_pop[1]
        original_solution = current_pop[2] + child_pop[2]
        Original_Pop.append(original_tour)
        Original_Pop.append(original_value)
        Original_Pop.append(original_solution)  # 拼接两个种群
        # 删除重复元素
        unique_elements = set()
        duplicate_indices = set()
        for i, element in enumerate(original_value):
            if element in unique_elements:
                duplicate_indices.add(i)
            else:
                unique_elements.add(element)

        original_value = [element for i, element in enumerate(original_value) if i not in duplicate_indices]
        original_tour = [element for i, element in enumerate(original_tour) if i not in duplicate_indices]
        original_solution = [element for i, element in enumerate(original_solution) if i not in duplicate_indices]
        # 排序，并取前十个
        # 将三个数组打包成元组列表
        zipped_lists = list(zip(original_value, original_tour, original_solution))
        # 根据数组一的顺序对元组列表进行排序
        sorted_zipped_lists = sorted(zipped_lists)
        # 解压排序后的元组列表，得到排序后的三个数组
        sorted_value, sorted_tour, sorted_solution = zip(*sorted_zipped_lists)
        # 将排序后的元组转换为列表
        sorted_value = list(sorted_value)
        sorted_tour = list(sorted_tour)
        sorted_solution = list(sorted_solution)
        pop_to_ls.append(sorted_tour)
        pop_to_ls.append(sorted_value)
        pop_to_ls.append(sorted_solution)

        return pop_to_ls
