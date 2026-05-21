import torch

from src.encoders.multimodal_encoder import MultimodalEncoder


def test_cross_modal_similarity_shapes():
    encoder = MultimodalEncoder(model_type="clip", model_name="openai/clip-vit-base-patch32")
    image_embeddings = torch.randn(2, 4)
    text_embeddings = torch.randn(3, 4)
    sim = encoder.cross_modal_similarity(image_embeddings, text_embeddings)
    assert sim.shape == (2, 3)
