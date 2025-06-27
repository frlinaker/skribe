# promptlearn

**promptlearn** is a drop-in extension for `scikit-learn` that brings the power of LLMs to existing machine learning pipelines.
It provides `scikit-learn`-compatible estimators powered by LLMs, such as:

- `PromptClassifier` – classifies data with LLM prompts
- `PromptRegressor` – regresses data with LLM prompts

The estimators support standard `fit()` and `predict()` methods, and can be used in the same way as any other `scikit-learn` estimator.

The noteworthy difference from traditional estimators is that the LLM is able to *automatically detect and exploit higher-level patterns* in the data,
for example

> the target is an XOR of x1 and x2

> the equation y=2x*3 is a good linear fit for this noisy data

> here is a simple human-readable decision tree that classifies this data well (followed by the actual tree)

The estimators detect these patterns during the `fit()` step, and then use this knowledge to output exact predictions in the `predict()` step.

Further, and more impressively, these systems can leverage the LLM's memory and infer data that is 'missing' from the input. For example, it is able to accurately return color information about country flags, being given as input only the name of a country. This is impossible for traditional machine learning models. Conceptually, this can be thought of as a `web-join`, i.e. the input is automatically joined with relevant information from the entire web, as captured by the LLM during its construction.

Another unique feature is that these estimators can learn from ZERO rows of data, as long as it is given the names of the input and output features. It can leverage the absorbed world knowledge of the LLM to this avail. An example of this is in [examples/zero_row_classifier.py](examples/zero_row_classifier.py) which is able to build a fully functioning classifier by just the column names 'country_name' and 'has_blue_in_flag'. This is not possible with traditional machine learning models.

---

## 📁 License

MIT © 2025 Fredrik Linaker
