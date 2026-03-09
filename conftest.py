# Root conftest.py
# These files are standalone scripts, not pytest test modules.
# They call sys.exit() at module level which breaks pytest collection.
collect_ignore = [
    "tests/test_fama_french.py",
    "tests/test_feature_integration.py",
    "tests/test_rate_limiter.py",
    "tests/test_sector_features.py",
    "tests/test_all_providers_e2e.py.OLD",
]
