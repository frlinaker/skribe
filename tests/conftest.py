import os

# Point the live-LLM test suite at a fast, cheap model so the pre-commit gate
# stays quick, while the shipped default (see promptlearn.base.DEFAULT_MODEL)
# remains the flagship model. Estimators resolve PROMPTLEARN_MODEL at
# construction time, so this takes effect for any test that builds an estimator
# without passing an explicit model. Export PROMPTLEARN_MODEL yourself to run
# the suite against a different model.
os.environ.setdefault("PROMPTLEARN_MODEL", "gpt-5.4-mini")
