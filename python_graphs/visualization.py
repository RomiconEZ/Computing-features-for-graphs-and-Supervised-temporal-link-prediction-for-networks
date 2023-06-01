import random

import pandas as pd
import python_graphs.base as graphs
import importlib

from python_graphs.model_training import get_performance

importlib.reload(graphs)

network_rus_name = 'Сеть'
cat_graph_rus_name = 'Категория'
nodes_rus_name = 'Вершины'
type_of_edge_rus_name = 'Тип ребер'
edges_rus_name = 'Ребра'
dens_rus_name = 'Плот.'
part_of_nodes_rus_name = 'Доля вершин'
scc_rus_name = 'КСС'
num_of_nodes_in_big_scc_rus_name = 'Вершины в наиб.КСС'
num_of_edges_in_big_scc_rus_name = 'Ребра в наиб.КСС'
radius_sb_rus_name = 'Радиус(ск)'
diameter_sb_rus_name = 'Диаметр(ск)'
procentile_sb_rus_name = '90проц.расст.(ск)'
radius_rnc_rus_name = 'Радиус(свв)'
diameter_rnc_rus_name = 'Диаметр(свв)'
procentile_rnc_rus_name = '90проц.расст.(свв)'
asst_fact_rus_name = 'Коэф.ассорт.'
avr_cl_fact_rus_name = 'Ср.кл.коэф.'
auc_rus_name = 'AUC'


def get_stats(network_info):
    tmpGraph = graphs.TemporalGraph(network_info['Path'])
    staticGraph = tmpGraph.get_static_graph(0., 1.)

    adjacency_dict_of_dicts = staticGraph.get_adjacency_dict_of_dicts()
    node1 = random.choice(list(adjacency_dict_of_dicts.keys()))
    node2 = random.choice(list(adjacency_dict_of_dicts[node1].keys()))

    snowball_sample_approach = graphs.SelectApproach(node1,node2)
    random_selected_vertices_approach = graphs.SelectApproach()
    sg_sb = snowball_sample_approach(staticGraph.get_largest_connected_component())
    sg_rsv = random_selected_vertices_approach(staticGraph.get_largest_connected_component())
    # ск - снежный ком
    # свв - случайный выбор вершин
    result = {}
    try:
        result[network_rus_name] = network_info['Label']
    except KeyError:
        result[network_rus_name] = None

    try:
        result[cat_graph_rus_name] = network_info['Category']
    except KeyError:
        result[cat_graph_rus_name] = None

    try:
        result[nodes_rus_name] = staticGraph.count_vertices()
    except Exception:
        result[nodes_rus_name] = None

    try:
        result[type_of_edge_rus_name] = network_info['Edge type']
    except KeyError:
        result[type_of_edge_rus_name] = None

    try:
        result[edges_rus_name] = staticGraph.count_edges()
    except Exception:
        result[edges_rus_name] = None

    try:
        result[dens_rus_name] = staticGraph.density()
    except Exception:
        result[dens_rus_name] = None

    try:
        result[part_of_nodes_rus_name] = staticGraph.share_of_vertices()
    except Exception:
        result[part_of_nodes_rus_name] = None

    try:
        result[scc_rus_name] = staticGraph.get_number_of_connected_components()
    except Exception:
        result[scc_rus_name] = None
    print('Получили число компонент связности')
    try:
        result[num_of_nodes_in_big_scc_rus_name] = staticGraph.get_largest_connected_component().count_vertices()
    except Exception:
        result[num_of_nodes_in_big_scc_rus_name] = None
    print('Получили число вершин в наибольшей компоненте связности')
    try:
        result[num_of_edges_in_big_scc_rus_name] = staticGraph.get_largest_connected_component().count_edges()
    except Exception:
        result[num_of_edges_in_big_scc_rus_name] = None

    try:
        result[radius_sb_rus_name] = staticGraph.get_radius(sg_sb)
    except Exception:
        result[radius_sb_rus_name] = None

    try:
        result[diameter_sb_rus_name] = staticGraph.get_diameter(sg_sb)
    except Exception:
        result[diameter_sb_rus_name] = None

    try:
        result[procentile_sb_rus_name] = staticGraph.percentile_distance(sg_sb)
    except Exception:
        result[procentile_sb_rus_name] = None

    try:
        result[radius_rnc_rus_name] = staticGraph.get_radius(sg_rsv)
    except Exception:
        result[radius_rnc_rus_name] = None

    try:
        result[diameter_rnc_rus_name] = staticGraph.get_diameter(sg_rsv)
    except Exception:
        result[diameter_rnc_rus_name] = None

    try:
        result[procentile_rnc_rus_name] = staticGraph.percentile_distance(sg_rsv)
    except Exception:
        result[procentile_rnc_rus_name] = None

    try:
        result[asst_fact_rus_name] = staticGraph.assortative_factor()
    except Exception:
        result[asst_fact_rus_name] = None
    print("Получили коэф. ассорт.")
    try:
        result[avr_cl_fact_rus_name] = staticGraph.average_cluster_factor()
    except Exception:
        result[avr_cl_fact_rus_name] = None
    print("Получили сред.класт.коэф.")
    try:
        result[auc_rus_name] = get_performance(tmpGraph, 0.67)
    except Exception:
        result[auc_rus_name] = None

    return result


def graph_features_tables(datasets_info: pd.DataFrame):
    table = pd.DataFrame([get_stats(network_info) for index, network_info in datasets_info.iterrows()]).sort_values(
        'Вершины')


    columns_to_include_to_feature_network_table_1 = [
        network_rus_name, cat_graph_rus_name, nodes_rus_name, type_of_edge_rus_name, edges_rus_name, dens_rus_name,
        part_of_nodes_rus_name
    ]
    columns_to_include_to_feature_network_table_2 = [
        network_rus_name, scc_rus_name, num_of_nodes_in_big_scc_rus_name,
        num_of_edges_in_big_scc_rus_name
    ]
    columns_to_include_to_feature_network_table_3 = [
        network_rus_name, radius_sb_rus_name, diameter_sb_rus_name,
        procentile_sb_rus_name, radius_rnc_rus_name, diameter_rnc_rus_name, procentile_rnc_rus_name,
    ]
    columns_to_include_to_feature_network_table_4 = [
        network_rus_name, asst_fact_rus_name, avr_cl_fact_rus_name,
    ]

    columns_to_include_to_auc_table = [
        network_rus_name, auc_rus_name,
    ]
    latex_feature_network_table_1 = table.to_latex(
        formatters={
            nodes_rus_name: lambda x: f'{x:,}',
            edges_rus_name: lambda x: f'{x:,}',
            dens_rus_name: lambda x: f'{x:.6f}',
            part_of_nodes_rus_name: lambda x: f'{x:.6f}',

        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Признаки для сетей, рассмотренных в ходе работы"
        ),
        label='Таблица: Признаки сетей',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_1
    )
    latex_feature_network_table_2 = table.to_latex(
        formatters={

            scc_rus_name: lambda x: f'{x:,}',
            num_of_nodes_in_big_scc_rus_name: lambda x: f'{x:,}',
            num_of_edges_in_big_scc_rus_name: lambda x: f'{x:,}',

        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Признаки для сетей, рассмотренных в ходе работы"
        ),
        label='Таблица: Признаки сетей',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_2
    )

    latex_feature_network_table_3 = table.to_latex(
        formatters={

            radius_sb_rus_name: lambda x: f'{x:.2f}',
            diameter_sb_rus_name: lambda x: f'{x:.2f}',
            procentile_sb_rus_name: lambda x: f'{x:.2f}',
            radius_rnc_rus_name: lambda x: f'{x:.2f}',
            diameter_rnc_rus_name: lambda x: f'{x:.2f}',
            procentile_rnc_rus_name: lambda x: f'{x:.2f}',

        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Признаки для сетей, рассмотренных в ходе работы"
        ),
        label='Таблица: Признаки сетей',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_3
    )
    latex_feature_network_table_4 = table.to_latex(
        formatters={

            asst_fact_rus_name: lambda x: f'{x:.2f}',
            avr_cl_fact_rus_name: lambda x: f'{x:.2f}',
        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Признаки для сетей, рассмотренных в ходе работы"
        ),
        label='Таблица: Признаки сетей',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_4
    )
    latex_auc_table = table.to_latex(
        formatters={
            auc_rus_name: lambda x: f'{x:.2f}',
        },
        column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
        index=False,
        caption=(
            "Точность предсказания появления ребер"
        ),
        label='Таблица: AUC',
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_auc_table
    )
    return (latex_feature_network_table_1, latex_feature_network_table_2, latex_feature_network_table_3,
            latex_feature_network_table_4, latex_auc_table)
