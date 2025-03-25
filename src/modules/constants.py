"""Common constants shared across modules."""

# Constants for loss types
LOSS_TYPES = {
    "TEXT": "text",
    "AUDIO": "audio",
    "GENDER": "gender",
    "FUSION": "fusion",
}

# Constants for embedding types
EMBEDS = {
    "TEXT": "text_embed",
    "AUDIO": "audio_embed",
    "FUSION": "fusion_embed",
    "ID": "fusion_embed",  # Default to fusion, can be overridden
    "CLASS": "class_preds"
}

# Update ID embedding if needed
def set_id_embedding(embed_feats):
    """Set which embedding to use for speaker ID."""
    global EMBEDS
    EMBEDS["ID"] = f"{embed_feats}_embed"
