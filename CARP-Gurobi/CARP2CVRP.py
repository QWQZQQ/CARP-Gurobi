import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

def carp_to_cvrp(G, demands, required_edges, depot, vehicle_capacity):
    """
    将CARP实例转化为CVRP实例（Longo等人方法，2r+1节点）
    参数:
        G: networkx无向图，节点为int，边为(i,j)，边属性'cost'
        demands: dict, (i,j): demand
        required_edges: list, 需要服务的边 (i,j)
        depot: int, 仓库节点编号
        vehicle_capacity: int, 车辆容量
    返回:
        cvrp_nodes: list, 节点列表，0为depot，其余为s_ij, s_ji
        cvrp_demands: dict, 节点->需求
        distance_matrix: np.ndarray, 距离矩阵（完全图）
        must_edges: list, 必须被选择的边(s_ij, s_ji)对
    """
    # 1. 预处理：为每个required_edge生成两个节点
    node_map = {}  # (i,j): (s_ij, s_ji)
    cvrp_nodes = [0]  # 0为depot
    cvrp_demands = {0: 0}
    idx = 1
    for (i, j) in required_edges:
        s_ij = idx
        s_ji = idx + 1
        node_map[(i, j)] = (s_ij, s_ji)
        cvrp_nodes.extend([s_ij, s_ji])
        # 可以任意分配需求, 这里平均分配
        d = demands[(i, j)]
        cvrp_demands[s_ij] = d / 2
        cvrp_demands[s_ji] = d / 2
        idx += 2

    n = len(cvrp_nodes)
    distance_matrix = np.full((n, n), np.inf)
    # depot编号为0
    for u in range(n):
        distance_matrix[u, u] = 0

    # 2. 计算距离
    # (a) depot到s_ij: 最短路(depot, i)
    for (i, j), (s_ij, s_ji) in node_map.items():
        dist_depot_i = nx.shortest_path_length(G, depot, i, weight='cost')
        dist_depot_j = nx.shortest_path_length(G, depot, j, weight='cost')
        distance_matrix[0, s_ij] = dist_depot_i
        distance_matrix[0, s_ji] = dist_depot_j
        distance_matrix[s_ij, 0] = dist_depot_i
        distance_matrix[s_ji, 0] = dist_depot_j

    # (b) s_ij, s_ji之间
    for (i, j), (s_ij, s_ji) in node_map.items():
        # 必须经过(s_ij, s_ji)或反向
        distance_matrix[s_ij, s_ji] = G[i][j]['cost']
        distance_matrix[s_ji, s_ij] = G[i][j]['cost']

    # (c) s_ij到s_kl (i,j)!=(k,l)
    for (i1, j1), (s_ij, s_ji) in node_map.items():
        for (i2, j2), (s_kl, s_lk) in node_map.items():
            if (i1, j1) == (i2, j2):
                continue
            # s_ij到s_kl: i到k的最短路
            d1 = nx.shortest_path_length(G, i1, i2, weight='cost')
            d2 = nx.shortest_path_length(G, i1, j2, weight='cost')
            d3 = nx.shortest_path_length(G, j1, i2, weight='cost')
            d4 = nx.shortest_path_length(G, j1, j2, weight='cost')
            distance_matrix[s_ij, s_kl] = d1
            distance_matrix[s_ij, s_lk] = d2
            distance_matrix[s_ji, s_kl] = d3
            distance_matrix[s_ji, s_lk] = d4

    # 3. 必须选中的边(s_ij, s_ji)
    must_edges = []
    for (i, j), (s_ij, s_ji) in node_map.items():
        must_edges.append((s_ij, s_ji))

    return cvrp_nodes, cvrp_demands, distance_matrix, must_edges



def solve_cvrp_gurobi(nodes, demands, distance_matrix, vehicle_capacity, must_edges, depot=0):
    """
    用Gurobi求解CVRP
    nodes: list, 节点编号（0为depot）
    demands: dict, 节点->需求
    distance_matrix: np.ndarray (n,n)
    vehicle_capacity: int
    must_edges: list of (i,j), 必须选中的边(如(s_ij, s_ji))
    depot: int, depot节点编号
    """
    n = len(nodes)
    # max_veh_num = len(must_edges)
    max_veh_num = 6
    # ======= 创建模型 =======
    model = gp.Model("CARP2CVRP")

    # ======= 定义变量 =======
    # 1. 弧选择变量（二进制）
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i][j] < np.inf:
                for k in range(max_veh_num):
                    x[i, j, k] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{i}_{j}_{k}"
                    )

    # 2. 车辆使用变量（二进制）
    y = {}
    for k in range(max_veh_num):
        y[k] = model.addVar(
            vtype=GRB.BINARY,
            name=f"y_{k}"
        )

    # 3. MTZ变量（用于防止子回路）
    u = {}
    for i in range(1,n):  # 包含仓库节点
        for k in range(max_veh_num):
            u[i, k] = model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"u_{i}_{k}",
                lb=demands.get(nodes[i], 0),  # 剩余容量最小为0
                ub=vehicle_capacity  # 最大为车辆容量
            )

    # ======= 目标函数 =======
    obj = gp.quicksum(
        distance_matrix[i][j] * x.get((i, j, k), 0)
        for i in range(n)
        for j in range(n)
        for k in range(max_veh_num)
        if i != j and (i, j, k) in x
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # ======= 关键约束 =======
    # 1. 每个非仓库节点必须被访问一次
    for i in range(1, n):  # 跳过depot（索引0）
        model.addConstr(
            gp.quicksum(
                x.get((i, j, k), 0)
                for j in range(n)
                for k in range(max_veh_num)
                if i != j and (i, j, k) in x
            ) == 1,
            f"visit_{i}"
        )

    # 2. 流平衡（每个非仓库节点的入度=出度）
    for k in range(max_veh_num):
        for h in range(1, n):  # 非仓库节点
            inflow = gp.quicksum(
                x.get((i, h, k), 0)
                for i in range(n)
                if i != h and (i, h, k) in x
            )
            outflow = gp.quicksum(
                x.get((h, j, k), 0)
                for j in range(n)
                if j != h and (h, j, k) in x
            )
            model.addConstr(
                inflow == outflow,
                f"flow_balance_{h}_{k}"
            )
    # 3. 车辆使用约束（从仓库的流出表示车辆被使用）
    for k in range(max_veh_num):
        # 从仓库出发的弧
        outflow = gp.quicksum(
            x.get((depot, j, k), 0)
            for j in range(1, n)
            if (depot, j, k) in x
        )
        # 回到仓库的弧
        inflow = gp.quicksum(
            x.get((i, depot, k), 0)
            for i in range(1, n)
            if (i, depot, k) in x
        )

        model.addConstr(
            outflow == y[k],
            f"veh_use_out_{k}"
        )
        model.addConstr(
            inflow == y[k],
            f"veh_use_in_{k}"
        )

    # 4. 容量约束
    for k in range(max_veh_num):
        model.addConstr(
            gp.quicksum(
                demands.get(nodes[i], 0) *
                gp.quicksum(
                    x.get((i, j, k), 0)
                    for j in range(n)
                    if i != j and (i, j, k) in x
                )
                for i in range(1, n)  # 非仓库节点
            ) <= vehicle_capacity * y[k],
            f"capacity_{k}"
        )

    # 5. 防止子回路的MTZ约束
    for k in range(max_veh_num):  # 遍历所有可能车辆
        for i in range(n):
            for j in range(1, n):  # 仅限客户节点
                if i != j and (i, j, k) in x:
                    if i == 0:  # 从仓库出发
                        model.addConstr(
                            u[j, k] <= vehicle_capacity - demands.get(nodes[j], 0),
                            f"depot_to_{j}_{k}"
                        )
                    else:  # 客户节点之间
                        model.addConstr(
                            u[j, k] <= u[i, k] - demands.get(nodes[j], 0) * x[i, j, k]
                            + vehicle_capacity * (1 - x[i, j, k]),
                            f"mtz_{i}_{j}_{k}"
                        )

    # 6. 关键约束：确保s_ij和s_ji连续访问
    # 必须在访问s_ij后立即访问s_ji（或反之），不能中间访问其他节点
    for (a, b) in must_edges:
        # 只允许两种连续访问：a->b 或 b->a
        model.addConstr(
            gp.quicksum(
                x.get((i, j, k), 0)
                for i in [a, b]
                for j in [a, b]
                for k in range(max_veh_num)
                if i != j and (i, j, k) in x
            ) == 1,
            f"must_edge_{a}_{b}"
        )


    # ======= 求解 =======
    model.setParam(GRB.Param.TimeLimit, 3600)  # 5分钟限时
    model.optimize()

    # ======= 提取解决方案 =======
    routes = []
    total_distance = 0

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        for i in range(n):
            for j in range(n):
                if i != j and (i,j,k) in x:
                    k = 0
                    val = x[i,j,k].X  # 获取变量值
                    if val > 0.5:  # 只打印值为1的变量
                        print(f"x[{i},{j},{k}] = {val:.0f} (距离: {distance_matrix[i][j]})")


        # 按车辆提取路径
        for k in range(max_veh_num):
            if y[k].X > 0.5:  # 车辆k被使用
                current = depot
                route = [depot]
                while True:
                    next_node = None
                    for j in range(n):
                        if current != j and (current, j, k) in x and x[current, j, k].X > 0.5:
                            next_node = j
                            break
                    if next_node is None or next_node == depot:
                        break
                    route.append(next_node)
                    current = next_node
                route.append(depot)
                routes.append(route)

                # 计算此路径的距离
                dist = 0
                for idx in range(len(route) - 1):
                    i, j = route[idx], route[idx + 1]
                    dist += distance_matrix[i][j]
                total_distance += dist

        print(f"最优解找到! 总距离: {model.ObjVal:.2f}")
        print(f"使用车辆数: {len(routes)}")
        for i, route in enumerate(routes):
            dist = sum(distance_matrix[route[j]][route[j + 1]] for j in range(len(route) - 1))
            print(f"车辆 {i + 1} 路线: {'->'.join(str(n) for n in route)} (距离: {dist:.2f})")
    else:
        print("未找到可行解")

    return model, routes, total_distance




def parse_gdb_file(file_path):
    """
    解析标准GDB格式的CARP实例文件（论文中Table 4使用的gdb1-23数据集）
    返回格式：
        - G: 带权图（包含cost/demand属性）
        - demands: 边需求字典
        - required_edges: 需服务的边列表
        - depot: 仓库节点编号（转换为0-based）
        - vehicle_capacity: 车辆容量
    """
    G = nx.Graph()
    demands = {}
    required_edges = []
    vehicle_capacity = None
    depot = None

    in_edge_list = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # 解析车辆容量（论文实验统一参数）
            if line.startswith("CAPACIDAD :"):
                vehicle_capacity = int(line.split(':')[1].strip())

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

                G.add_edge(u - 1, v - 1, cost=cost)
                required_edges.append((u - 1, v - 1))
                demands[(u - 1, v - 1)] = demand



    return G, demands, required_edges, depot, vehicle_capacity








# 示例用法
if __name__ == '__main__':

    file_path = "./gdb/gdb1.dat"
    G, demands, required_edges, depot, vehicle_capacity = parse_gdb_file(file_path)

    cvrp_nodes, cvrp_demands, distance_matrix, must_edges = carp_to_cvrp(
        G, demands, required_edges, depot, vehicle_capacity
    )


    solve_cvrp_gurobi(
        cvrp_nodes,
        cvrp_demands,
        distance_matrix,
        vehicle_capacity,
        must_edges,
        depot=0
    )