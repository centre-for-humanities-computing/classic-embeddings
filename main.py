import os
from pathlib import Path

from dash_extensions.enrich import Dash
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.model import Model
from gensim.models import KeyedVectors


def get_models(path: str) -> dict[str, Model]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    models = {}
    for model_name in model_names:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        keyed_vectors = KeyedVectors.load(str(model_path))
        models[model_name] = Model.from_keyed_vectors(keyed_vectors)
    return models


models = get_models(path="dat")
blueprint, register_pages = create_dashboard(models, fuzzy_search=True)
app = get_dash_app(
    blueprint=blueprint, name=__name__, use_pages=True, assets_folder="assets/"
)
register_pages()

server = app.server


if __name__ == "__main__":
    app.run_server(debug=True, port=8080, host="0.0.0.0")
