import torch

from classic_algos.nn import (
    TilesMLP,
    GatedAttention,
    MLP,
)


class HE2RNA(torch.nn.Module):
    
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        d_model_attention: int = 256,
    ):
        super(HE2RNA, self).__init__()
        
        self.tiles_emb = TilesMLP(
            in_features,
            out_features=d_model_attention,
        )

        self.attention_layer = GatedAttention(
            d_model=d_model_attention,
        )

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            activation=torch.nn.Sigmoid(),
        )
    
    def forward(self, x, mask):
        tiles_emb = self.tiles_emb(
            x[..., 3:], mask)
        scaled_tiles_emb, attention_weights = self.attention_layer(
            tiles_emb, mask)
        
        logits = self.mlp(scaled_tiles_emb)
        
        return logits
