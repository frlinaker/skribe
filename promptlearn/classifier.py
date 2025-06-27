import os

from sklearn.base import BaseEstimator, ClassifierMixin
import openai

class PromptClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.prompt_template = """You are a principal data scientist. Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, equation, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns. Data: {data}"""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = openai.OpenAI()

    def fit(self, X, y):
        # Format data into table form for LLM
        n_features = X.shape[1]
        headers = [f"x{i+1}" for i in range(n_features)] + ["target"]

        # Join into tab-separated rows
        data_rows = ["\t".join(headers)]
        for xi, yi in zip(X, y):
            row = list(map(str, xi)) + [str(yi)]
            data_rows.append("\t".join(row))

        formatted_data = "\n".join(data_rows)
        self.prompt_ = self.prompt_template.format(data=formatted_data)
        print(self.prompt_)

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": self.prompt_}],
            temperature=0.0,
            max_tokens=800
        )

        self.classification_prompt_ = response.choices[0].message.content.strip()
        print(self.classification_prompt_)

        return self

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        # Convert input vector into named feature rows
        feature_names = [f"x{i+1}" for i in range(len(x))]
        feature_string = ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, x))

        full_prompt = (
            self.classification_prompt_ + "\n\n"
            f"Given: {feature_string}\n"
            "What is the predicted target class?\n"
            "Respond only with a number (0, 1, 2, etc)."
        )

        print(full_prompt)

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            max_tokens=10
        )

        prediction = response.choices[0].message.content.strip()

        print(f"Prediction: {prediction}")

        return int(prediction)
