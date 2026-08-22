Audio fixtures come from `librosa.example(...)`, which downloads and caches
them on first use, so nothing binary is tracked here. `tests/test_cli.py` and
`tests/test_train.py` generate their own synthetic audio instead.
