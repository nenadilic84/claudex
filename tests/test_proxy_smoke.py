# Placeholder: A real test would mock the environment or downstream HTTP calls.
def test_proxy_importable():
    import claudex.proxy
    assert hasattr(claudex.proxy, "app")
