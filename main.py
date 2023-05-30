import os
from itertools import islice
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from dash_extensions.enrich import Dash
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.model import StaticEmbeddings
from gensim.models import KeyedVectors


def get_models(path: str) -> dict[str, StaticEmbeddings]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    models = {}
    for model_name in model_names:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        keyed_vectors = KeyedVectors.load(str(model_path))
        models[model_name] = StaticEmbeddings.from_keyed_vectors(keyed_vectors)
    return models


# Setting template to use the SBL font
pio.templates["greek"] = go.layout.Template(layout=dict(font_family="SBL Greek"))
pio.templates.default = "greek"

models = get_models(path="dat")
blueprint, register_pages = create_dashboard(models, fuzzy_search=True)
app = get_dash_app(
    blueprint=blueprint, name=__name__, use_pages=True, assets_folder="assets/"
)
register_pages()

server = app.server


if __name__ == "__main__":
    app.run_server(debug=False, port=8080, host="0.0.0.0")
