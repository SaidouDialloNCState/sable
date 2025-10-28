def test_imports():
    import strategy_backtester as sb
    assert isinstance(sb.__version__, str)
