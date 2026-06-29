from sklearn.neural_network import MLPClassifier

def create_mlp_model():
    """
    Returns a configured MLPClassifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=300,
        alpha=0.01,
        random_state=42
    )
