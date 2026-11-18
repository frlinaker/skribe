import os

# Point the live-LLM test suite at a fast, cheap model so the pre-commit gate
# stays quick, while the shipped default (see skribe.base.DEFAULT_MODEL)
# remains the flagship model. Estimators resolve SKRIBE_MODEL at
# construction time, so this takes effect for any test that builds an estimator
# without passing an explicit model. Export SKRIBE_MODEL yourself to run
# the suite against a different model.
os.environ.setdefault("SKRIBE_MODEL", "gpt-5.4-mini")

# LLM-output persistence is on by default (see skribe/persist.py); keep the
# suite from littering the working tree. tests/test_persist.py opts back in.
os.environ.setdefault("SKRIBE_LOG_DIR", "off")
