import trend_estimation as td


def test_metric_tables():
    table = td.error_metrics_table([1,2,3], [1,2,4], model_name='m')
    assert {'model', 'reference', 'MAE', 'RMSE'}.issubset(table.columns)
    comp = td.compare_error_tables({'obs':[1,2,3]}, {'m':[1,2,4]})
    assert {'target', 'model', 'RMSE'}.issubset(comp.columns)
