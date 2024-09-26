import bisect
import copy
import random
import time
from math import inf

from Individual import Individual


class Population:
    def __init__(self, popsize, problem):
        self.popsize = popsize
        self.problem = problem
        self.dismatrix = problem.dismatrix
        self.alpha = problem.alpha
        self.kappa = problem.kappa
        self.idv = Individual()
        self.crossover_rate = 0.8
        self.mutation_rate = 0.05

    def creat_pop(self):
        P = []
        solution = self.problem.initial_tsp
        P.append(solution)
        l = len(solution) - 2
        # 进行随机扰动
        while len(P) < self.popsize:  # 大小为popsize的种群
            idx1 = random.randint(1, l)
            idx2 = random.randint(1, l)
            if idx1 > idx2:
                temp = idx1
                idx1 = idx2
                idx2 = temp
            change_sol = copy.deepcopy(solution)  # 复制求解器求解的初始解
            r = random.random()  # 生成一个随机数
            if r < 0.5:
                change_sol[idx1:idx2 + 1] = change_sol[idx1:idx2 + 1][::-1]
            else:
                elements_to_shuffle = change_sol[idx1:idx2 + 1]
                random.shuffle(elements_to_shuffle)
                change_sol[idx1:idx2 + 1] = elements_to_shuffle
            if self.check_sol(change_sol, P):  # check == true 原来种群中没有这个解，则加入
                P.append(change_sol)

        return P

    def check_sol(self, sol, P):
        for idv in P:
            if sol == idv:
                return False
        return True

    def next_pop(self, Pop):
        P_parent = []  # 父代种群，由锦标赛选出，不含TSP-D方案和fitness
        parent_tour = []
        parent_value = []
        parent_solution = []
        P_child = []  # 子代种群，包含TSP-D方案和fitness
        child_tour = []
        child_value = []
        child_solution = []
        # 二进制锦标赛选择父代种群Pparent，与原来的种群大小一致
        for i in range(self.popsize):
            [parent, p_value, p_solution] = self.binary_tournament(Pop)
            parent_tour.append(parent)
            parent_value.append(p_value)
            parent_solution.append(p_solution)
        P_parent.append(parent_tour)
        P_parent.append(parent_value)
        P_parent.append(parent_solution)

        # 父种群中每个元素进行交叉变异
        for j in range(len(P_parent[0])):
            individual = P_parent[0][j]
            if random.random() < self.crossover_rate:  # 若选择的染色体达到交叉概率，则随机从父代中选择另一个染色体与之进行交叉
                random_individual = random.choice(P_parent[0])
                [child_tour, child_value, child_solution] = self.crossover(individual, random_individual, child_tour, child_value, child_solution)
            else:
                if random.random() < self.mutation_rate:  # 若达到变异概率，将该个体进行变异
                    mutation_child = self.mutation(individual)  # 整个big_tour进行变异
                    # [child_tour, child_value, child_solution] = self.ep_assigned([mutation_child], child_tour, child_value, child_solution)
                    [child_tour, child_value, child_solution] = self.assigned_noadjust([mutation_child], child_tour, child_value, child_solution)
                else:
                    bisect.insort(child_value, P_parent[1][j])  # 将元素插入到从小到大排列的数组中
                    index = child_value.index(P_parent[1][j])
                    child_tour.insert(index, P_parent[0][j])
                    child_solution.insert(index, P_parent[2][j])

        P_child.append(child_tour)
        P_child.append(child_value)
        P_child.append(child_solution)

        return P_child

    def binary_tournament(self, population):
        # 二进制锦标赛
        random_number1 = random.randint(0, len(population[0])-1)  # 生成一个随机数
        parent_a = population[0][random_number1]  # 随机选择的染色体
        a_value = population[1][random_number1]  # 该染色体的value值
        a_solution = population[2][random_number1]  # 该染色体对应的具体分配方案
        # 定位到该染色体的下标
        random_number2 = random.randint(0, len(population[0])-1)  # 生成一个随机数
        parent_b = population[0][random_number2]  # 随机选择的染色体
        b_value = population[1][random_number2]  # 该染色体的value值
        b_solution = population[2][random_number2]

        if a_value < b_value:
            return [parent_a, a_value, a_solution]
        else:
            return [parent_b, b_value, b_solution]

    def evaluate(self, parent):
        truck = parent[1][0]
        drone = parent[1][1]
        total_distance = 0
        i = 0

        while i < len(truck) - 1:  # 遍历卡车路径的节点
            trucki = truck[i]
            # 首先判断无人机架次中是否有以这个为起点的架次
            while drone:
                remove = inf
                for n in range(0,len(drone)):
                    if drone[n][0] == trucki:
                        remove = 0
                        sortie = (self.dismatrix[drone[n][0]][drone[n][1]] + self.dismatrix[drone[n][1]][drone[n][2]]) / self.alpha
                        if drone[n][2] == 0:
                            index = len(truck) - 1
                            truck_cost = 0
                            for idx in range(i, index):
                                truck_cost += self.dismatrix[truck[idx]][truck[idx + 1]]
                            total_distance += max(sortie, truck_cost)
                            i = index
                            # # 将计算过的架次从无人机数组中移除
                            # drone = drone[:n]+drone[n+1:]
                            remove = n
                            break
                        else:
                            index = truck.index(drone[n][2])
                            truck_cost = 0
                            for idx in range(i, index):
                                truck_cost += self.dismatrix[truck[idx]][truck[idx + 1]]
                            total_distance += max(sortie, truck_cost)
                            i = index
                            # # 将计算过的架次从无人机数组中移除
                            # drone = drone[:n]+drone[n+1:]
                            remove = n

                if remove == inf:
                    total_distance += self.dismatrix[truck[i]][truck[i + 1]]
                    # i += 1
                break

            if i == len(truck)-1:
                break
            elif i != len(truck)-1 and drone == []:
                total_distance += self.dismatrix[truck[i]][truck[i + 1]]
                i = i + 1
            elif i != len(truck)-1 and drone != [] and remove != inf:
                drone = drone[:remove] + drone[remove + 1:]
            elif i != len(truck) - 1 and drone != [] and remove == inf:
                i += 1

        return total_distance

    def crossover(self, parent_a, parent_b, tour, value, solution):
        a = copy.deepcopy(parent_a)
        a = a[1:-1]
        b = copy.deepcopy(parent_b)
        b = b[1:-1]
        # 交叉位置
        y = random.randint(0, len(a))
        # 记录交叉项
        fragment1 = a[y:]
        fragment2 = b[y:]
        aa = []
        for i in a[:y]:
            while i in fragment2:
                i = fragment1[fragment2.index(i)]
            aa.append(i)
        children1 = aa + fragment2
        children1.insert(0,0)
        children1.append(0)
        # 第二层染色体的卡车无人机通过分配来决定
        # [final_pop, final_pop_value, final_pop_solution] = self.ep_assigned([children1], tour, value, solution)  # 交叉后的进行分配
        [final_pop, final_pop_value, final_pop_solution] = self.assigned_noadjust([children1], tour, value, solution)

        return [final_pop, final_pop_value, final_pop_solution]

    def mutation(self, child):
        # 从四种方式中随机选择一种方式进行变异
        # 1.基于位置的变异：随机地产生两个变异位，然后将第二个变异位上的基因移动到第一个变异位之前。
        # 2.基于位置的变异：随机地产生两个变异位，然后将第一个变异位上的基因移动到第二个变异位之后。
        # 3.基于次序的变异：该方法随机地产生两个变异位，然后交换这两个变异位上的基因。
        # 4.翻转切片变异：该方法随机产生两个变异位，作为起始位置和结束位置，将两位置之间的基因翻转。
        mutation_child = []
        i = random.randint(1, 4)
        if i == 1:
            mutation_child = self.position_based1(child)
        elif i == 2:
            mutation_child = self.position_based2(child)
        elif i == 3:
            mutation_child = self.order_based(child)
        elif i == 4:
            mutation_child = self.slice_mutation(child)
        return mutation_child

    def position_based1(self, child):
        child_cut = child[1:-1]
        size = len(child_cut)

        # 生成两个不重复的随机变异位
        mutation_points = random.sample(range(1, size), 2)

        # 获取两个变异位的索引
        mutation_point1 = min(mutation_points)
        mutation_point2 = max(mutation_points)

        # 移动第二个变异位上的基因到第一个变异位之前
        gene_to_move = child_cut.pop(mutation_point2)
        child_cut.insert(mutation_point1, gene_to_move)

        child_cut.insert(0, 0)
        child_cut.append(0)

        return child_cut

    def position_based2(self, child):
        child_cut = child[1:-1]
        size = len(child_cut)

        # 生成两个不重复的随机变异位
        mutation_points = random.sample(range(1, size), 2)

        # 获取两个变异位的索引
        mutation_point1 = min(mutation_points)
        mutation_point2 = max(mutation_points)

        # 移动第一个变异位上的基因到第二个变异位之后
        gene_to_move = child_cut.pop(mutation_point1)
        child_cut.insert(mutation_point2 + 1, gene_to_move)

        child_cut.insert(0, 0)
        child_cut.append(0)

        return child_cut

    def order_based(self, child):
        child_cut = child[1:-1]
        size = len(child_cut)

        # 生成两个不重复的随机变异位
        mutation_points = random.sample(range(size), 2)

        # 获取两个变异位的索引
        mutation_point1 = mutation_points[0]
        mutation_point2 = mutation_points[1]

        # 交换两个变异位上的基因
        child_cut[mutation_point1], child_cut[mutation_point2] = child_cut[mutation_point2], child_cut[mutation_point1]
        child_cut.insert(0, 0)
        child_cut.append(0)

        return child_cut

    def slice_mutation(self, child):
        child_cut = child[1:-1]
        size = len(child_cut)

        # 生成两个不重复的随机变异位
        mutation_points = random.sample(range(size), 2)

        # 获取两个变异位的索引
        mutation_point1 = min(mutation_points)
        mutation_point2 = max(mutation_points)

        # 翻转两个变异位之间的基因
        child_cut[mutation_point1:mutation_point2 + 1] = child_cut[mutation_point1:mutation_point2 + 1][::-1]
        child_cut.insert(0, 0)
        child_cut.append(0)

        return child_cut

    def assigned(self, pop, final_pop, final_pop_value, final_pop_solution):
        f_pop = final_pop
        f_pop_value = final_pop_value
        # 存放无人机与卡车分配方案
        f_pop_solution = final_pop_solution

        for i in range(len(pop)):  # 遍历pop中的染色体，decoding每一个
            l = len(pop[i])
            cut_node_length = 50
            # 分治规模改变
            a = (l - 1) % cut_node_length
            if a == 0:
                cut_num = (l - 1) // cut_node_length
            if a > 0:
                cut_num = ((l - 1) // cut_node_length) + 1
            truck_tour = []
            drone_sorties = []
            f_value = 0
            end_sortie = []
            end_value = 0
            for cut in range(cut_num):
                # 将染色体表示的tsp路径分为10个节点一段
                if a == 0:
                    sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]
                if 0 < a < 3:
                    if cut == cut_num-1:
                        break
                    if cut == cut_num-2:
                        sub = pop[i][(cut * cut_node_length):]
                    if cut != cut_num-1 and cut != cut_num-2:
                        sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]
                if a >= 3:
                    if cut == cut_num-1:
                        sub = pop[i][(cut * cut_node_length):]
                    if cut != cut_num - 1:
                        sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]

                # 每个子路径求解
                [[sub_tour], sub_value, sub_solution] = self.idv.assign_drone(sub, self.dismatrix, self.alpha, self.kappa)
                # 如果是第一个路径
                if cut == 0:
                    t1 = 0
                    t2 = 0
                    end_sortie = sub_solution[0][1][len(sub_solution[0][1])-1]
                    t1 = (self.dismatrix[end_sortie[0]][end_sortie[1]] + self.dismatrix[end_sortie[1]][end_sortie[2]]) / self.alpha
                    index = sub_solution[0][0].index(end_sortie[0])
                    end_truck = sub_solution[0][0][index:]
                    for node in range(len(end_truck)-1):
                        t2 = t2 + self.dismatrix[end_truck[node]][end_truck[node+1]]
                    end_value = max(t1,t2)

                    f_value = f_value + sub_value
                    truck_tour = sub_solution[0][0]
                    drone_sorties = drone_sorties + sub_solution[0][1]

                # 如果是最后一个路径
                if cut == cut_num-1 and cut_num != 1:
                    t1 = 0
                    t2 = 0
                    start_sortie = sub_solution[0][1][0]
                    t1 = (self.dismatrix[start_sortie[0]][start_sortie[1]] + self.dismatrix[start_sortie[1]][
                        start_sortie[2]]) / self.alpha
                    index = sub_solution[0][0].index(start_sortie[2])
                    start_truck = sub_solution[0][0][0:index + 1]
                    for node in range(len(start_truck) - 1):
                        t2 = t2 + self.dismatrix[start_truck[node]][start_truck[node + 1]]
                    start_value = max(t1, t2)

                    # 以下是优化部分------
                    if start_sortie[2] == 0:
                        combine = pop[i][(pop[i].index(end_sortie[0])):]
                    else:
                        combine = pop[i][(pop[i].index(end_sortie[0])):(pop[i].index(start_sortie[2]) + 1)]
                    combine_value_origin = start_value + end_value
                    [[combine_tour], combine_value, combine_solution] = self.idv.assign_drone(combine, self.dismatrix,
                                                                                              self.alpha, self.kappa)
                    # 比较修改过后的value值,如果比原来的小，则用修改后的方案代替原来的
                    if combine_value < combine_value_origin:
                        f_value = f_value + (sub_value - (combine_value_origin - combine_value))
                        # 卡车路径
                        if start_sortie[2] == 0:
                            truck_tour = truck_tour[:truck_tour.index(end_sortie[0])] + combine_solution[0][0]
                        else:
                            truck_tour = truck_tour[:truck_tour.index(end_sortie[0])] + combine_solution[0][0] + sub_solution[0][0][(sub_solution[0][0].index(start_sortie[2]) + 1):]

                        drone_sorties = drone_sorties[:-1] + combine_solution[0][1] + sub_solution[0][1][1:]

                    else:
                        f_value = f_value + sub_value
                        truck_tour = truck_tour[:-1] + sub_solution[0][0]
                        drone_sorties = drone_sorties + sub_solution[0][1]

                # 其他情况
                if cut > 0 and cut < cut_num-1:
                    t1 = 0
                    t2 = 0
                    start_sortie = sub_solution[0][1][0]
                    t1 = (self.dismatrix[start_sortie[0]][start_sortie[1]] + self.dismatrix[start_sortie[1]][start_sortie[2]]) / self.alpha
                    index = sub_solution[0][0].index(start_sortie[2])
                    start_truck = sub_solution[0][0][0:index+1]
                    for node in range(len(start_truck)-1):
                        t2 = t2 + self.dismatrix[start_truck[node]][start_truck[node+1]]
                    start_value = max(t1,t2)

                    # 以下是优化部分------
                    if start_sortie[2] == 0:
                        combine = pop[i][(pop[i].index(end_sortie[0])):]
                    else:
                        combine = pop[i][(pop[i].index(end_sortie[0])):(pop[i].index(start_sortie[2]) + 1)]
                    combine_value_origin = start_value + end_value
                    [[combine_tour], combine_value, combine_solution] = self.idv.assign_drone(combine, self.dismatrix, self.alpha, self.kappa)
                    # 比较修改过后的value值,如果比原来的小，则用修改后的方案代替原来的
                    if combine_value < combine_value_origin:
                        f_value = f_value + (sub_value - (combine_value_origin - combine_value))
                        # 卡车路径
                        if start_sortie[2] == 0:
                            truck_tour = truck_tour[:truck_tour.index(end_sortie[0])] + combine_solution[0][0]
                        else:
                            truck_tour = truck_tour[:truck_tour.index(end_sortie[0])] + combine_solution[0][0] + sub_solution[0][0][(sub_solution[0][0].index(start_sortie[2]) + 1):]

                        drone_sorties = drone_sorties[:-1] + combine_solution[0][1] + sub_solution[0][1][1:]
                    else:
                        f_value = f_value + sub_value
                        truck_tour = truck_tour[:-1] + sub_solution[0][0]
                        drone_sorties = drone_sorties + sub_solution[0][1]

                    end_sortie = drone_sorties[len(drone_sorties) - 1]
                    t1 = (self.dismatrix[end_sortie[0]][end_sortie[1]] + self.dismatrix[end_sortie[1]][end_sortie[2]]) / self.alpha
                    index = truck_tour.index(end_sortie[0])
                    end_truck = truck_tour[index:]
                    t2 = 0
                    for node in range(len(end_truck) - 1):
                        t2 = t2 + self.dismatrix[end_truck[node]][end_truck[node + 1]]
                    end_value = max(t1, t2)

            # 表示solution
            solution = []
            solution.append(truck_tour)
            solution.append(drone_sorties)

            f_pop_value.append(f_value)
            f_pop.append(pop[i])
            f_pop_solution.append(solution)

        return [f_pop, f_pop_value, f_pop_solution]

    def assigned_noadjust(self, pop, final_pop, final_pop_value, final_pop_solution):
        f_pop = final_pop
        f_pop_value = final_pop_value
        # 存放无人机与卡车分配方案
        f_pop_solution = final_pop_solution

        for i in range(len(pop)):  # 遍历pop中的染色体，decoding每一个
            l = len(pop[i])
            cut_node_length = 10
            # 分治规模改变
            a = (l - 1) % cut_node_length
            if a == 0:
                cut_num = (l - 1) // cut_node_length
            if a > 0:
                cut_num = ((l - 1) // cut_node_length) + 1
            truck_tour = []
            drone_sorties = []
            f_value = 0
            end_sortie = []
            end_value = 0
            for cut in range(cut_num):
                # 将染色体表示的tsp路径分为10个节点一段
                if a == 0:
                    sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]
                if 0 < a < 3:
                    if cut == cut_num-1:
                        break
                    if cut == cut_num-2:
                        sub = pop[i][(cut * cut_node_length):]
                    if cut != cut_num-1 and cut != cut_num-2:
                        sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]
                if a >= 3:
                    if cut == cut_num-1:
                        sub = pop[i][(cut * cut_node_length):]
                    if cut != cut_num - 1:
                        sub = pop[i][(cut * cut_node_length):((cut + 1) * cut_node_length + 1)]

                # 每个子路径求解
                [[sub_tour], sub_value, sub_solution] = self.idv.assign_drone(sub, self.dismatrix, self.alpha, self.kappa)
                # 如果是第一个路径
                if cut == 0:


                    f_value = f_value + sub_value
                    truck_tour = sub_solution[0][0]
                    drone_sorties = drone_sorties + sub_solution[0][1]
                else:
                    f_value = f_value + sub_value
                    truck_tour = truck_tour[:-1] + sub_solution[0][0]
                    drone_sorties = drone_sorties + sub_solution[0][1]


            # 表示solution
            solution = []
            solution.append(truck_tour)
            solution.append(drone_sorties)

            f_pop_value.append(f_value)
            f_pop.append(pop[i])
            f_pop_solution.append(solution)

        return [f_pop, f_pop_value, f_pop_solution]

    def ep_assigned(self, pop, final_pop, final_pop_value, final_pop_solution):
        f_pop = final_pop
        f_pop_value = final_pop_value
        # 存放无人机与卡车分配方案
        f_pop_solution = final_pop_solution

        for i in range(len(pop)):  # 遍历pop中的染色体，decoding每一个
            l = len(pop[i])
            sub = pop[i]

                # 每个子路径求解
            # [[tour], value, solution] = self.idv.assign_drone(sub, self.dismatrix, self.alpha, self.kappa)
            [tour, value, solution] = self.idv.exact_partitioning2(sub, self.dismatrix, self.alpha, self.kappa)

            f_pop_value.append(value)
            f_pop.append(pop[i])
            f_pop_solution.append(solution)

        return [f_pop, f_pop_value, f_pop_solution]

    def is_value_in_list(self,value, valuelist):
        return value in valuelist

    def environmental_selection(self, pop, popsize):
        sorted_indices = sorted(range(len(pop[1])), key=lambda x: pop[1][x])
        sorted_value = [pop[1][i] for i in sorted_indices]
        sorted_solution = [pop[0][i] for i in sorted_indices]
        sorted_value = sorted_value[:popsize]
        sorted_solution = sorted_solution[:popsize]
        pop = [sorted_solution, sorted_value]
        return pop
