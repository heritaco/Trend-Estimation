def test_import():
    import trend_estimation as td
    assert hasattr(td, 'PenalizedTrend')
