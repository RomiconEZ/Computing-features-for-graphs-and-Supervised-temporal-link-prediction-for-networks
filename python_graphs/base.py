from typing import Optional
from pydantic import BaseModel
from dataclasses import dataclass
import numpy as np


class Node(BaseModel):
    number: int

    def __lt__(self, other: 'Node'):
        return self.number < other.number

    def __eq__(self, other: 'Node'):
        return self.number == other.number


class Edge(BaseModel):
    start_node: Node
    end_node: Node
    timestamp: int

    def __lt__(self, other: 'Edge'):
        return max(self.end_node, self.start_node) < max(other.end_node, other.start_node)

    def __eq__(self, other: 'Edge'):
        return self.start_node == other.start_node and self.end_node == other.end_node and self.timestamp == other.timestamp

    def get_max_node(self):
        return max(self.end_node, self.start_node)


class TemporalGraph:
    edge_list: list[Edge]

    def __init__(self, path: str):
        self.edge_list = list()
        with open(path) as raw_data:
            raw_data.readline()
            raw_data.readline()

            list_of_items = raw_data.read().split('\n')
            list_of_items.pop(-1)
            for item in list_of_items:
                item = item.split(" ")
                if int(item[0]) == int(item[1]): # не учитываем петли
                    continue
                self.edge_list.append(
                    Edge(
                        start_node=Node(number=int(item[0])),
                        end_node=Node(number=int(item[1])),
                        timestamp=int(item[-1]),
                    )
                )

        self.edge_list.sort(key=lambda x: x.timestamp)

    def get_static_graph(self, l: float, r: float) -> 'StaticGraph':
        left_index = int(l * len(self.edge_list))
        right_index = int(r * len(self.edge_list))
        sg = StaticGraph()
        for i in range(left_index, right_index):
            sg.add_edge(self.edge_list[i])
        return sg

    def get_max_timestamp(self):
        return max(self.edge_list, key=lambda x: x.timestamp).timestamp

    def get_min_timestamp(self):
        return min(self.edge_list, key=lambda x: x.timestamp).timestamp


@dataclass
class StaticGraph:
    num_of_edge: int = 0
    num_of_node: int = 0
    adjacency_dict_of_dicts: dict[int, dict[int, [int]]] = None  # Матрица смежности
    largest_connected_component: Optional['StaticGraph'] = None
    number_of_connected_components: Optional[int] = None

    def __init__(self):
        self.adjacency_dict_of_dicts = dict()
        self.largest_connected_component = None
        self.number_of_connected_components = None

    def add_node(self, node: Node) -> int:
        # добавим вершину
        self.adjacency_dict_of_dicts[node.number] = dict()
        self.num_of_node += 1
        return self.num_of_node

    def add_edge(self, edge: Edge) -> int:
        start_node_number = edge.start_node.number
        end_node_number = edge.end_node.number

        # если вершин не существует, добавим их, и сохраним их индексы
        if start_node_number not in self.adjacency_dict_of_dicts.keys():
            self.add_node(edge.start_node)

        if end_node_number not in self.adjacency_dict_of_dicts.keys():
            self.add_node(edge.end_node)

        # свапнем вершины, если start_node_index > end_node_index
        if start_node_number > end_node_number:
            start_node_number, end_node_number = end_node_number, start_node_number

        if end_node_number not in self.adjacency_dict_of_dicts[start_node_number].keys():  # если ребро пришло первый раз

            self.adjacency_dict_of_dicts[start_node_number][end_node_number] = [edge.timestamp]
            self.adjacency_dict_of_dicts[end_node_number][start_node_number] = []
            self.num_of_edge += 1

        elif edge.timestamp not in self.adjacency_dict_of_dicts[start_node_number][
            end_node_number]:  # проверка на полный дубликат
            self.adjacency_dict_of_dicts[start_node_number][end_node_number] += [edge.timestamp]
            self.num_of_edge += 1
        return self.num_of_edge

    def count_vertices(self) -> int:
        return self.num_of_node

    def get_adjacency_dict_of_dicts(self) -> dict:
        return self.adjacency_dict_of_dicts


    def count_edges(self) -> int:
        return self.num_of_edge

    def density(self) -> float:
        cnt_vert: int = self.count_vertices()
        return self.count_edges() / (cnt_vert * (cnt_vert - 1))

    def __find_size_of_connected_component(self, used: dict, start_vertice) -> int:
        queue = list()
        used[start_vertice] = True
        queue.append(start_vertice)

        size = 0

        while len(queue) > 0:
            v = queue.pop(0)
            size += 1
            for to in self.adjacency_dict_of_dicts[v].keys():
                if to not in used.keys():
                    used[to] = True
                    queue.append(to)

        return size

    def __find_largest_connected_component(self, used: dict, start_vertice):
        # Обойдём всю компоненту слабой связности и запишем её как отдельный граф
        queue = list()
        used[start_vertice] = True
        queue.append(start_vertice)

        while len(queue) > 0:
            v = queue.pop(0)
            for to in self.adjacency_dict_of_dicts[v].keys():
                if to not in used.keys():
                    used[to] = True
                    queue.append(to)

                min_node = min(v, to)
                max_node = max(v, to)
                edge_ts = self.adjacency_dict_of_dicts[min_node][max_node]
                for i in edge_ts:
                    start_node = Node(number=min_node)
                    end_node = Node(number=max_node)
                    timestamp = i
                    self.largest_connected_component.add_edge(
                        Edge(start_node=start_node, end_node=end_node, timestamp=timestamp))

    def __update_number_of_connected_components_and_largest_connected_component(self):
        # Запустим DFS от каждой ещё не посещённой вершины, получая компоненты слабой связности
        # Заодно считаем количество этих компонент и максимальную по мощности компоненту слабой связности
        used = dict()
        vertice: int = 0
        self.number_of_connected_components = 0
        max_component_size: int = 0
        for v in self.adjacency_dict_of_dicts.keys():
            if v not in used.keys():
                self.number_of_connected_components += 1
                component_size = self.__find_size_of_connected_component(used, v)
                if component_size > max_component_size:
                    max_component_size = component_size
                    vertice = v

        # Обновляем посещенность вершин для обработки максимальной по мощности компоненты
        # слабой связности
        used.clear()

        # Нашли максимальную по мощности компоненту слабой связности, запишем её в поле
        self.largest_connected_component = StaticGraph()
        self.__find_largest_connected_component(used, vertice)

    def get_largest_connected_component(self) -> 'StaticGraph':
        # если максимальную по мощность компоненту слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.largest_connected_component

    def get_number_of_connected_components(self) -> int:
        # если число компонент слабой связности не нашли, найдём
        if self.largest_connected_component is None:
            self.__update_number_of_connected_components_and_largest_connected_component()

        return self.number_of_connected_components

    def share_of_vertices(self) -> float:
        return self.get_largest_connected_component().count_vertices() / self.count_vertices()

    def get_radius(self, method: 'SelectApproach') -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        shortest_paths: dict[int, dict[int, int]] = dict()
        for i in sample_graph.adjacency_dict_of_dicts.keys():
            shortest_paths[i] = dict()
            for j in sample_graph.adjacency_dict_of_dicts[i].keys():
                shortest_paths[i][j] = 1
        for k in shortest_paths.keys():
            for i in shortest_paths[k].keys():
                for j in shortest_paths[k].keys():
                    if j not in shortest_paths[i].keys():
                        shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]
                    else:
                        shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        radius = np.inf
        for i in shortest_paths.keys():
            eccentricity = 0
            for j in shortest_paths[i].keys():
                eccentricity = max(eccentricity, shortest_paths[i][j])
            if eccentricity > 0:
                radius = min(radius, eccentricity)
        return radius

    def get_diameter(self, method: 'SelectApproach') -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        shortest_paths: dict[int, dict[int, int]] = dict()
        for i in sample_graph.adjacency_dict_of_dicts.keys():
            shortest_paths[i] = dict()
            for j in sample_graph.adjacency_dict_of_dicts[i].keys():
                shortest_paths[i][j] = 1
        for k in shortest_paths.keys():
            for i in shortest_paths[k].keys():
                for j in shortest_paths[k].keys():
                    if j not in shortest_paths[i].keys():
                        shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]
                    else:
                        shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        diameter = 0
        for i in shortest_paths.keys():
            for j in shortest_paths[i].keys():
                diameter = max(diameter, shortest_paths[i][j])
        return diameter

    def percentile_distance(self, method: 'SelectApproach', percentile: int = 90) -> int:
        # Находим подграф с помощью выбранного метода
        sample_graph: StaticGraph = method

        # Алгоритм Флойда-Уоршелла
        # cnt_verts = sample_graph.count_vertices()
        shortest_paths: dict[int, dict[int, int]] = dict()
        for i in sample_graph.adjacency_dict_of_dicts.keys():
            shortest_paths[i] = dict()
            for j in sample_graph.adjacency_dict_of_dicts[i].keys():
                shortest_paths[i][j] = 1
        for k in shortest_paths.keys():
            for i in shortest_paths[k].keys():
                for j in shortest_paths[k].keys():
                    if j not in shortest_paths[i].keys():
                        shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]
                    else:
                        shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j])

        dists = []
        for i in shortest_paths.keys():
            for j in shortest_paths[i].keys():
                dists.append(shortest_paths[i][j])
        dists.sort()
        return dists[int(percentile / 100 * (len(dists) - 1))]

    def average_cluster_factor(self) -> float:
        cnt_verts = self.get_largest_connected_component().count_vertices()
        result = 0
        for i in self.get_largest_connected_component().adjacency_dict_of_dicts.keys():
            i_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[i])
            if i_degree < 2:
                continue
            l_u = 0
            for j in self.get_largest_connected_component().adjacency_dict_of_dicts[i].keys():
                for k in self.get_largest_connected_component().adjacency_dict_of_dicts[i].keys():
                    if k in self.get_largest_connected_component().adjacency_dict_of_dicts[j]:
                        l_u += 1

            result += l_u / (i_degree * (i_degree - 1))
        return result / cnt_verts

    def assortative_factor(self) -> float:
        re = 0
        r1 = 0
        r2 = 0
        r3 = 0
        for u in self.get_largest_connected_component().adjacency_dict_of_dicts.keys():
            u_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[u])
            r1 += u_degree
            r2 += u_degree ** 2
            r3 += u_degree ** 3
            for v in self.get_largest_connected_component().adjacency_dict_of_dicts[u].keys():
                v_degree = len(self.get_largest_connected_component().adjacency_dict_of_dicts[v])
                re += u_degree * v_degree

        return (re * r1 - (r2 * r2)) / (r3 * r1 - (r2 * r2))


@dataclass
class SelectApproach:
    start_node1_number: Optional[int]
    start_node2_number: Optional[int]

    def __init__(self, s_node1_number: int = None, s_node2_number: int = None):

        if s_node1_number is not None and s_node2_number is not None:
            if s_node1_number > s_node2_number:
                s_node1_number, s_node2_number = s_node2_number, s_node1_number
        self.start_node1_number = s_node1_number
        self.start_node2_number = s_node2_number

    def snowball_sample(self, graph: StaticGraph) -> StaticGraph:
        queue = list()
        start_node1_number = self.start_node1_number
        start_node2_number = self.start_node2_number

        # добавляем две вершины в очередь для BFS
        queue.append(start_node1_number)
        queue.append(start_node2_number)
        cnt_verts = graph.count_vertices()

        size = min(500, cnt_verts)

        sample_graph = StaticGraph()  # новый граф, который должны получить в результате

        used = dict()
        used[start_node1_number] = True
        used[start_node2_number] = True

        size -= 2

        while len(queue) > 0:  # BFS
            v = queue.pop(0)

            for i in graph.adjacency_dict_of_dicts[v]:
                if i not in used.keys():
                    if size > 0:
                        used[i] = True
                        size -= 1
                        queue.append(i)
                    else:
                        continue

                min_node = min(v, i)
                max_node = max(v, i)

                edge_ts = graph.adjacency_dict_of_dicts[min_node][max_node]
                for i in edge_ts:
                    start_node = Node(number=min_node)
                    end_node = Node(number=max_node)
                    timestamp = i
                    sample_graph.add_edge(
                        Edge(start_node=start_node, end_node=end_node, timestamp=timestamp))

        return sample_graph

    def random_selected_vertices(self, graph: StaticGraph) -> StaticGraph:
        remaining_vertices = list(graph.adjacency_dict_of_dicts.keys())  # множество оставшихся вершин
        size = min(500, graph.count_vertices())
        sample_graph = StaticGraph()  # новый граф, который должны получить в результате

        for _ in range(size):
            # выберем новую вершину для добавления в граф
            new_vertice = remaining_vertices[np.random.randint(0, len(remaining_vertices))]

            remaining_vertices.remove(new_vertice)
            sample_graph.add_node(Node(number=new_vertice))
            for vertice in sample_graph.adjacency_dict_of_dicts.keys():
                # если вершины смежны в исходном графе, то добавим рёбра
                if new_vertice in graph.adjacency_dict_of_dicts[vertice].keys():
                    min_node = min(vertice, new_vertice)
                    max_node = max(vertice, new_vertice)

                    edge_ts = graph.adjacency_dict_of_dicts[min_node][max_node]
                    for i in edge_ts:
                        start_node = Node(number=min_node)
                        end_node = Node(number=max_node)
                        timestamp = i
                        sample_graph.add_edge(
                            Edge(start_node=start_node, end_node=end_node, timestamp=timestamp))

        return sample_graph

    def __call__(self, graph: StaticGraph):
        if self.start_node1_number is None:
            return self.random_selected_vertices(graph)
        return self.snowball_sample(graph)
