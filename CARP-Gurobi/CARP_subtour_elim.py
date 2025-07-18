import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# ======= 输入数据 =======
# 节点数量（通过边自动识别）
nodes = set()
# required_edges = [
#     # (起点, 终点, 费用, 需求)
#     (0, 1, 12, 8),
#     (1, 2, 6, 5),
#     (1, 3, 10, 3),
#     (2, 3, 7, 6),
#     (3, 4, 5, 4),
#     (0, 4, 10, 7)
# ]

# required_edges = [
#     (0, 1, 6, 4),
#     (0, 2, 8, 5),
#     (0, 3, 9, 6),
#     (1, 4, 7, 3),
#     (1, 5, 10, 4),
#     (2, 5, 5, 2),
#     (2, 6, 11, 5),
#     (3, 6, 8, 3),
#     (3, 7, 9, 6),
#     (4, 5, 6, 2),
#     (4, 8, 7, 4),
#     (5, 8, 5, 3),
#     (5, 9, 12, 7),
#     (6, 7, 7, 4),
#     (6, 9, 10, 6),
#     (7, 8, 6, 2),
#     (7, 9, 8, 4),
#     (8, 9, 9, 5),
#     (1, 2, 5, 3),
#     (2, 3, 6, 2),
# ]


def parse_gdb_file(file_path):
    # 初始化变量
    capacidad = None
    deposito = None
    required_edges = []
    in_edge_list = False
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            # 提取CAPACIDAD（车辆容量）
            if line.startswith("CAPACIDAD :"):
                capacity = int(line.split(':')[1].strip())

            # 提取DEPOSITO（仓库节点）
            elif line.startswith("LISTA_ARISTAS_REQ"):
                in_edge_list = True
            elif line.startswith("DEPOSITO"):
                in_edge_list = False
                depot = int(line.split(":")[1].strip()) - 1
            elif in_edge_list and line.startswith("("):
                # 解析边的信息
                parts = line.split("coste")
                edge_part = parts[0].strip().strip("() ")
                cost_part = parts[1].split("demanda")[0].strip()
                demand_part = parts[1].split("demanda")[1].strip()
                u, v = map(int, edge_part.split(","))
                cost = int(cost_part)
                demand = int(demand_part)
                required_edges.append((u - 1, v - 1,  cost, demand))

    return capacity, depot, required_edges

# file_path = "../测试集/graph_v30e50/graph_v30e50_1.dat"
file_path = "./gdb/gdb1.dat"
Q, depot, required_edges = parse_gdb_file(file_path)
print(f"车辆容量 (CAPACIDAD): {Q}")
print(f"仓库节点 (DEPOSITO): {depot} (原始文件中的{depot + 1})")
print("\n需求边列表 (LISTA_ARISTAS_REQ):")

for edge in required_edges:
    print(f"({edge[0]}, {edge[1]}) - 费用: {edge[2]}, 需求: {edge[3]}")

# 自动确定节点集合
for u, v, cost, demand in required_edges:
    nodes.add(u)
    nodes.add(v)
n = len(nodes)  # 节点数量

# depot = 0  # 仓库节点
# # 车辆最大容量
# Q = 20

# 估计最大可能需要的车辆数
# max_vehicles = len(required_edges)  # 最多每条边一辆车
max_vehicles = 5 # 最多每条边一辆车


# 创建有向服务边列表（两个方向）
directed_edges = []
# 创建费用字典（移动或服务时使用）
cost_dict = {}
for u, v, cost, demand in required_edges:
    # 添加正向边
    directed_edges.append((u, v, cost, demand))
    cost_dict[(u, v)] = cost
    cost_dict[(v, u)] = cost  # 双向同费用

    # 添加反向边
    directed_edges.append((v, u, cost, demand))

# 原始无向边集合（用于服务约束）
original_undirected_edges = set()
for u, v, cost, demand in required_edges:
    original_undirected_edges.add(frozenset({u, v}))

# ======= 创建模型 =======
model = gp.Model("CARP_Full_Required")

# ======= 定义变量 =======
# 1. 边服务变量：x[i,j,k] = 1 表示车辆k服务边(i,j)（有向）
x = model.addVars(
    [(i, j, k) for i, j, cost, demand in directed_edges
     for k in range(max_vehicles)],
    vtype=GRB.BINARY,
    name="x"
)

# 2. 车辆移动变量：y[i,j,k] = n 表示车辆k从节点i移动到节点j n次（非服务移动）
# 注意：只能在存在的边上移动
y = model.addVars(
    [(i, j, k) for i, j, cost, demand in directed_edges
     for k in range(max_vehicles)],
    vtype=GRB.INTEGER,
    lb=0,
    name="y"
)

# 3. 车辆使用变量
z = model.addVars(max_vehicles, vtype=GRB.BINARY, name="z")

# 4. 载重变量（用于容量约束）
# load = model.addVars(
#     [(i, k) for i in nodes for k in range(max_vehicles)],
#     lb=0, ub=Q,
#     name="load"
# )

# ======= 目标函数 =======
# 最小化总费用 = 服务费用 + 非服务移动费用
total_cost = gp.quicksum(
    cost * x[i, j, k] for i, j, cost, demand in directed_edges
    for k in range(max_vehicles)
) + gp.quicksum(
    cost * y[i, j, k] for i, j, cost, demand in directed_edges
    for k in range(max_vehicles)
)

model.setObjective(total_cost, GRB.MINIMIZE)


# ======= 约束条件 =======

# 1. 每条需要服务的无向边必须被恰好一辆车服务一次（可以任选方向）
for edge in original_undirected_edges:
    u, v = tuple(edge)
    model.addConstr(
        gp.quicksum(x[i, j, k]
                    for i, j, cost, demand in directed_edges
                    for k in range(max_vehicles)
                    if {i, j} == edge) == 1,
        f"service_edge_{u}_{v}"
    )

# 2. 车辆必须从仓库出发并返回仓库
for k in range(max_vehicles):
    # 车辆离开仓库
    model.addConstr(
        gp.quicksum(y[i, j, k]
                    for i, j, cost, demand in directed_edges
                    if i == depot) +
        gp.quicksum(x[i, j, k]
                    for i, j, cost, demand in directed_edges
                    if i == depot) == z[k],
        f"depart_depot_{k}"
    )

    # 车辆返回仓库
    model.addConstr(
        gp.quicksum(y[i, j, k]
                    for i, j, cost, demand in directed_edges
                    if j == depot) +
        gp.quicksum(x[i, j, k]
                    for i, j, cost, demand in directed_edges
                    if j == depot) == z[k],
        f"return_depot_{k}"
    )

# 3. 流平衡约束：对每个节点，进入的边数等于离开的边数
for k in range(max_vehicles):
    for node in nodes:
        # 进入节点的流（非服务移动 + 服务边进入）
        inflow = gp.quicksum(
            y[i, j, k] for i, j, cost, demand in directed_edges
            if j == node
        ) + gp.quicksum(
            x[i, j, k] for i, j, cost, demand in directed_edges
            if j == node
        )

        # 离开节点的流（非服务移动 + 服务边离开）
        outflow = gp.quicksum(
            y[i, j, k] for i, j, cost, demand in directed_edges
            if i == node
        ) + gp.quicksum(
            x[i, j, k] for i, j, cost, demand in directed_edges
            if i == node
        )

        model.addConstr(inflow == outflow, f"flow_balance_{node}_{k}")

# 4. 容量约束
for k in range(max_vehicles):
    # 车辆总载重不超过容量
    total_load = gp.quicksum(
        demand * x[i, j, k] for i, j, cost, demand in directed_edges
    )
    model.addConstr(total_load <= Q * z[k], f"capacity_{k}")
    model.addConstr(total_load >= 1 * z[k], f"capacity_lower_{k}") # 至少1单位需求（假设需求最小为1）
    model.addConstr(z[k] >= total_load / (Q + 1),f"active_vehicle_{k}")

# 5. 车辆使用逻辑约束
# for k in range(max_vehicles):
#     # 如果车辆k服务任何边，则车辆k被使用
#     model.addConstr(
#         gp.quicksum(x[i, j, k] for i, j, cost, demand in directed_edges) <=
#         len(directed_edges) * z[k],
#         f"vehicle_usage_{k}"
#     )

# 6. 服务与移动互斥约束：一条边不能同时被服务和非服务方式通过
# for i, j, cost, demand in directed_edges:
#     for k in range(max_vehicles):
#         model.addConstr(
#             x[i, j, k] + y[i, j, k] <= 1,
#             f"service_move_mutex_{i}_{j}_{k}"
#         )

# 7. MTZ载重更新约束
# for k in range(max_vehicles):
#     for i, j, cost, demand in directed_edges:
#         model.addConstr(
#             load[j, k] >= load[i, k] + demand * x[i, j, k]
#                           - Q * (1 - (x[i, j, k] + y[i, j, k])),
#             f"load_update_{i}_{j}_{k}"
#         )

# 8. 仓库载重为0
# for k in range(max_vehicles):
#     model.addConstr(load[depot, k] == 0, f"depot_load_{k}")

# 9. 对称性破除：强制车辆按顺序使用
for k in range(max_vehicles - 1):
    model.addConstr(z[k] >= z[k + 1], f"symmetry_{k}")


# ======= 子回路消除回调函数 =======
def subtour_elimination(model, where):
    if where == GRB.Callback.MIPSOL:
        # 获取当前解的变量值
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)

        # 遍历每辆车
        for k in range(max_vehicles):
            # 如果车辆未使用则跳过
            if model.cbGetSolution(model._z[k]) < 0.5:
                continue

            # 构建车辆k的路径图
            # G_k = nx.MultiDiGraph()
            G_k = nx.DiGraph()
            # 添加当前车辆的所有活动
            for i, j, cost, demand in directed_edges:
                # 添加服务边
                if x_val.get((i, j, k), 0) > 0.5:
                    G_k.add_edge(i, j, type='service')

                # 添加移动边（可能多次）
                y_count = int(y_val.get((i, j, k), 0))
                for idx in range(y_count):
                    # G_k.add_edge(i, j, type='move',weight = idx)
                    G_k.add_edge(i, j, type='move')

            # 如果没有边则跳过
            if G_k.number_of_edges() == 0:
                continue

            # connected_num = nx.number_strongly_connected_components(G_k)
            # model.cbLazy(connected_num <= 1)
            # 检测强连通分量
            for comp in nx.strongly_connected_components(G_k):
                # 如果分量包含仓库或是整个图则跳过
                if depot in comp or len(comp) == G_k.number_of_nodes():
                    continue

                # 创建诱导子图
                S = set(comp)
                SG = G_k.subgraph(S)

                # 如果图是强连通（即闭环）则添加消除约束
                if nx.is_strongly_connected(SG):

                    # 计算离开S的切割能力
                    sub_outgoing = gp.quicksum(
                        x[i, j, k] + y[i, j, k]
                        for i, j, cost, demand in directed_edges
                        if i in S and j not in S
                    )

                    # 添加懒惰约束：必须有边离开这个子集
                    model.cbLazy(sub_outgoing >= 1,
                                 f"subtour_elim_vehicle{k}_nodes{'_'.join(map(str, sorted(S)))}")

                    # sub_edges = gp.quicksum(
                    #     x[i, j, k] + y[i, j, k]
                    #     for i, j, cost, demand in directed_edges
                    #     if i in S and j in S
                    # )
                    # model.cbLazy(sub_edges <= len(S) - 1,
                    #              f"subtour_elim_vehicle{k}_nodes{'_'.join(map(str, sorted(S)))}")
                    # is_depot = gp.quicksum(
                    #         y[i, j, k]
                    #         for i, j, cost, demand in directed_edges
                    #         if (i in S and j in S) and (i == depot or j == depot)
                    #     ) + gp.quicksum(
                    #     x[i, j, k]
                    #     for i, j, cost, demand in directed_edges
                    #     if (i in S and j in S) and (i == depot or j == depot)
                    # )
                    #
                    # model.cbLazy(is_depot >= 1,
                    #              f"subtour_elim_vehicle{k}_nodes{'_'.join(map(str, sorted(S)))}")




def subtour_elimination_num(model, where):
    if where == GRB.Callback.MIPSOL:
        # 获取当前解的变量值
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)

        # 遍历每辆车
        for k in range(max_vehicles):
            # 如果车辆未使用则跳过
            if model.cbGetSolution(model._z[k]) < 0.5:
                continue

            # 构建车辆k的路径图
            # G_k = nx.MultiDiGraph()
            G_k = nx.DiGraph()
            # 添加当前车辆的所有活动
            for i, j, cost, demand in directed_edges:
                # 添加服务边
                if x_val.get((i, j, k), 0) > 0.5:
                    G_k.add_edge(i, j, type='service')

                # 添加移动边（可能多次）
                y_count = int(round(y_val.get((i, j, k), 0)))
                for idx in range(y_count):
                    # G_k.add_edge(i, j, type='move',weight = idx)
                    G_k.add_edge(i, j, type='move')

            # 如果没有边则跳过
            if G_k.number_of_edges() == 0:
                continue

            # 检测所有不包含depot的强连通子图
            invalid_components = [
                comp for comp in nx.strongly_connected_components(G_k)
            ]

            # 如果存在至少两个非法子图，添加约束
            if len(invalid_components) >= 2:
                model.cbLazy(0 >= 1, f"invalid_subtours_vehicle_{k}")
                return  # 终止当前回调





def subtour_elimination_connectivity_based(model, where):
    if where == GRB.Callback.MIPSOL:
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)

        # 预先计算有效边集合
        # valid_edges = set((i, j) for i, j, _, _ in directed_edges)

        for k in range(max_vehicles):
            if model.cbGetSolution(model._z[k]) < 0.5:
                continue

            # 构建车辆路径图
            G_k = nx.MultiDiGraph()
            # G_k = nx.DiGraph()
            G_k.add_node(depot)
            for i, j, cost, demand in directed_edges:
                if x_val.get((i, j, k), 0) > 0.5:
                    G_k.add_edge(i, j, type='service')
                y_count = int(round(y_val.get((i, j, k), 0)))
                for idx in range(y_count):
                    G_k.add_edge(i, j, type='move', weight=idx)
                    # G_k.add_edge(i, j, type='move')

            if G_k.number_of_edges() == 0:
                continue

            # 计算可达集合（使用强连通分量）
            try:
                # 正向可达
                nodes_reachable_from_depot = set()
                for comp in nx.strongly_connected_components(G_k):
                    if depot in comp:
                        nodes_reachable_from_depot |= comp

                # 反向可达
                G_rev = G_k.reverse()
                nodes_can_reach_depot = set()
                for comp in nx.strongly_connected_components(G_rev):
                    if depot in comp:
                        nodes_can_reach_depot |= comp
            except Exception:
                nodes_reachable_from_depot = {depot}
                nodes_can_reach_depot = {depot}

            # 处理强连通分量
            for comp in nx.strongly_connected_components(G_k):
                if depot in comp:
                    continue

                S = set(comp)

                # 单节点处理
                if len(S) == 1:
                    node = next(iter(S))
                    connection = gp.quicksum(
                        model._x[i, j, k] + model._y[i, j, k]
                        for i, j, _, _ in directed_edges
                        if (i == node and j in nodes_can_reach_depot) or
                        (j == node and i in nodes_reachable_from_depot)
                    )
                    model.cbLazy(connection >= 1)
                    continue

                # 子回路处理
                incoming_to_S = gp.quicksum(
                    model._x[i, j, k] + model._y[i, j, k]
                    for i, j, _, _ in directed_edges
                    if i in nodes_reachable_from_depot and j in S
                )

                outgoing_from_S = gp.quicksum(
                    model._x[i, j, k] + model._y[i, j, k]
                    for i, j, _, _ in directed_edges
                    if i in S and j in nodes_can_reach_depot
                )

                # 宽松约束

                model.cbLazy(
                    incoming_to_S + outgoing_from_S >= 1,
                    f"connectivity_veh{k}_nodes{'_'.join(map(str, sorted(S)))}"
                )
                # model.cbLazy(
                #     incoming_to_S >= 1,
                #     f"incoming_connect_{k}_nodes{'_'.join(map(str, sorted(S)))}"
                # )
                # model.cbLazy(
                #     outgoing_from_S >= 1,
                #     f"outgoing_connect_{k}_nodes{'_'.join(map(str, sorted(S)))}"
                # )

# 存储变量引用用于回调
model._x = x
model._y = y
model._z = z
# ======= 求解设置 =======
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 7200)
model.setParam("MIPGap", 0.00)

# model.Params.BestObjStop = 316
model.Params.LazyConstraints = 1  # 启用懒惰约束

# 求解
model.optimize(subtour_elimination_connectivity_based) # subtour_elimination_connectivity_based

# ======= 结果解析 =======
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"\n目标值（总费用）: {model.ObjVal:.2f}")

    # 统计使用的车辆数
    used_vehicles = sum(1 for k in range(max_vehicles) if z[k].X > 0.5)
    print(f"使用车辆数: {used_vehicles}")

    # 输出每辆车的路径/
    for k in range(max_vehicles):
        if z[k].X > 0.5:
            print(f"\n=== 车辆 {k + 1} ===")

            # 找到服务的边
            served_edges = []
            moved_edges = []
            total_demand = 0
            for i, j, cost, demand in directed_edges:
                if x[i, j, k].X > 0.5:
                    served_edges.append((i, j, demand))
                    total_demand += demand
                if y[i, j, k].X > 0.5:
                    for _ in range(int(y[i, j, k].X)):
                        moved_edges.append((i, j))

            print(f"服务的边: {served_edges}")
            print(f"移动的边: {moved_edges}")
            print(f"总载重: {total_demand}/{Q}")

            # 重构路径（简化版）
            path = []
            # 收集所有活动：服务移动和非服务移动
            for i, j, cost, demand in directed_edges:
                if y[i, j, k].X > 0.5:
                    for _ in range(int(round(y[i, j, k].X))):  # 添加多次
                        path.append((i, j, f"移动(费用:{cost})"))
                if x[i, j, k].X > 0.5:
                    path.append((i, j, f"服务(需求:{demand}, 费用:{cost})"))

            if path:
                print("动作序列:")
                # 按起点排序模拟路径顺序（简化逻辑）
                current = depot
                for _ in range(len(path)):
                    next_move = next(
                        (move for move in path if move[0] == current),
                        None
                    )
                    if next_move:
                        i, j, desc = next_move
                        print(f"  {i} → {j} - {desc}")
                        path.remove(next_move)
                        current = j
                    else:
                        break
            print(
                f"车辆总费用: {sum(cost for i, j, cost, demand in directed_edges if (x[i, j, k].X or y[i, j, k].X) > 0.5):.2f}")
    print(f"\n目标值（总费用）: {model.ObjVal:.2f}")

    # 验证解的有效性
    # print(f"\n=== 解的验证 ===")
    # for edge in original_undirected_edges:
    #     u, v = tuple(edge)
    #     served_count = sum(
    #         1 for i, j, cost, demand in directed_edges
    #         for k in range(max_vehicles)
    #         if {i, j} == edge and x[i, j, k].X > 0.5
    #     )
    #     print(f"边{tuple(edge)}: 被服务{served_count}次")

else:
    print("未找到可行解")
    print(f"求解状态: {model.status}")
    # 输出不可行约束
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")
        print("不可行的约束已写入model.ilp")


# # 创建图并验证节点
# G = nx.Graph()
# for u, v, cost, demand in required_edges:
#     G.add_edge(u, v, cost=cost, demand=demand)
#
# print("图中的节点:", sorted(G.nodes()))  # 应输出 [0,1,2,3,4]
#
# # 生成位置字典（包含所有节点）
# pos = nx.spring_layout(G, seed=42)
# pos[0] = (0, 0)  # 固定仓库位置
#
# # 绘制图形
# plt.figure(figsize=(10, 6))
# nx.draw(
#     G, pos,
#     node_color=['gold' if n == 0 else 'lightblue' for n in G.nodes()],
#     node_size=800,
#     with_labels=True,
#     labels={n: 'Depot' if n == 0 else str(n) for n in G.nodes()},
#     font_weight='bold'
# )
#
# # 添加边标签
# edge_labels = {(u, v): f"{G[u][v]['cost']}/{G[u][v]['demand']}"
#                for u, v in G.edges()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
# plt.title("CARP Problem Network")
# plt.axis('off')
# plt.show()

