import os

# Point the live-LLM test suite at a fast, cheap model so the pre-commit gate
# stays quick, while the shipped default (see skribe.base.DEFAULT_MODEL)
# remains the flagship model. Estimators resolve SKRIBE_MODEL at
# construction time, so this takes effect for any test that builds an estimator
# without passing an explicit model. Export SKRIBE_MODEL yourself to run
# the suite against a different model.
os.environ.setdefault("SKRIBE_MODEL", "gpt-5.4-mini")
