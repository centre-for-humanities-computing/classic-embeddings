import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from embedding_explorer import Card
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from gensim.models import KeyedVectors


def get_cards(path: str) -> list[Card]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    cards: list[Card] = []
    for model_name in model_names:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        kv = KeyedVectors.load(str(model_path))
        card = Card(model_name, corpus=kv.index_to_key, embeddings=kv.vectors)
        cards.append(card)
    return cards


# Setting template to use the SBL font
pio.templates["greek"] = go.layout.Template(layout=dict(font_family="SBL Greek"))
pio.templates.default = "greek"

cards = get_cards(path="dat")
blueprint, register_pages = create_dashboard(cards)
app = get_dash_app(
    blueprint=blueprint, name=__name__, use_pages=True, assets_folder="assets/"
)
register_pages()

server = app.server


if __name__ == "__main__":
    app.run_server(debug=False, port=8080, host="0.0.0.0")
