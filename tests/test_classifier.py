from promptlearn import PromptClassifier

def test_init():
    clf = PromptClassifier(prompt_template="Example: {{title}} {{tldr}}", llm_client=lambda x: "1")
    assert clf.prompt_template.startswith("Example")
