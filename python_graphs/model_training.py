import pandas as pd
import python_graphs.base as graphs
from python_graphs.feature_formation import feature_for_absent_edges, form_start_end_node_df
from sklearn import model_selection, pipeline, preprocessing, linear_model, metrics
import gc


def train_test_split_temporal_graph(edge_list: list, split_ratio: float):
    '''
    Разделение выборки на части формирования признаков и на части предсказания
    '''
    edge_list_feature_build_part = edge_list[:int(len(edge_list) * split_ratio)]
    edge_list_prediction_part = edge_list[len(edge_list_feature_build_part):]
    return edge_list_feature_build_part, edge_list_prediction_part


def add_label_column(ft_bld_edge_df: pd.DataFrame, pred_adj_dict: dict[int, dict[int, [int]]]):
    ft_bld_edge_df['label'] = ft_bld_edge_df.apply(
        lambda row: 1 if (pred_adj_dict.get(row['start_node']) is not None
                          and pred_adj_dict[row['start_node']].get(row['end_node']) is not None) else 0, axis=1)


def get_performance(temporalG: graphs.TemporalGraph, split_ratio: float):
    Edge_feature = feature_for_absent_edges(temporalG.get_static_graph(0, split_ratio).get_adjacency_dict_of_dicts(),
                                            temporalG.get_min_timestamp(),
                                            temporalG.get_max_timestamp())

    add_label_column(Edge_feature, temporalG.get_static_graph(split_ratio, 1).get_adjacency_dict_of_dicts())

    X = Edge_feature.drop(['label', 'start_node', 'end_node'], axis=1)
    y = Edge_feature['label']

    # Edge_feature.to_csv('Edge_feature.csv')

    del Edge_feature
    gc.collect()

    X_train, X_test, y_train, y_test = (
        model_selection.train_test_split(X, y, random_state=42))

    pipe = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(max_iter=10000, n_jobs=-1,
                                        random_state=42)
    )
    print("Начало обучения лог. регр.")
    pipe.fit(X_train, y_train)

    auc = metrics.roc_auc_score(
        y_true=y_test, y_score=pipe.predict_proba(X_test)[:, 1])

    return auc
