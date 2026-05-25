import trend_estimation as td


def test_difference_matrix_shape():
    D = td.difference_matrix(10, 2)
    assert D.shape == (8, 10)
