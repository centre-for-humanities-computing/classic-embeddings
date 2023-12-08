import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.cards import Card, ClusteringCard, NetworkCard
from gensim.models import KeyedVectors

SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def get_word_embedding_cards(path: str) -> list[NetworkCard]:
    """Get all models in a directory."""
    model_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    cards: list[NetworkCard] = []
    for model_name in model_names:
        model_path = Path(path).joinpath(model_name, "model.gensim")
        kv = KeyedVectors.load(str(model_path))
        card = NetworkCard(model_name, corpus=kv.index_to_key, embeddings=kv.vectors)
        cards.append(card)
    return cards


def get_corpus_card(vectors_path: str, corpus_name: str) -> ClusteringCard:
    keyed_embeddings = KeyedVectors.load(str(vectors_path))
    metadata = fetch_metadata(SHEET_URL)
    metadata = metadata.dropna(subset="document_id")
    embeddings = []
    has_embedding = []
    for doc_id in metadata.document_id:
        try:
            embeddings.append(keyed_embeddings[doc_id])
            has_embedding.append(True)
        except KeyError:
            has_embedding.append(False)
    metadata = metadata[has_embedding]
    embeddings = np.stack(embeddings)
    return ClusteringCard(
        f"Clustering: {corpus_name}",
        embeddings=embeddings,
        metadata=metadata,
        hover_name="work",
        hover_data=[
            "author",
            "group",
            "geografi",
            "årstal",
            "genre_first",
            "genre_second",
            "gender",
        ],
    )


# Setting template to use the SBL font
pio.templates["greek"] = go.layout.Template(layout=dict(font_family="SBL Greek"))
pio.templates.default = "greek"

network_cards = get_word_embedding_cards("dat")
whole_corpus_card = get_corpus_card(
    "dat/lemmatized_corpus.vectors.gensim", "Whole Corpus"
)
important_corpus_card = get_corpus_card(
    "dat/important_works.vectors.gensim", "Important Works"
)
cards: list[Card] = [*network_cards, whole_corpus_card, important_corpus_card]
blueprint, register_pages = create_dashboard(cards)
app = get_dash_app(
    blueprint=blueprint, name=__name__, use_pages=True, assets_folder="assets/"
)
register_pages()

server = app.server


if __name__ == "__main__":
    app.run_server(debug=False, port=8080, host="0.0.0.0")
