import torch
import torch.nn as nn
from torch import Tensor


class DBRL(nn.Module):
    """
    Device Behavior Representation Learning (DBRL)

    Stage 1 implementation:
        - Temporal behavior encoder
        - Tensor organization

    Graph learning and graph reasoning will be added later.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gru_layers = gru_layers

        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        #
        # Project GRU hidden state to the representation
        # dimension expected by downstream HDEG modules.
        #
        self.embedding_projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def _encode_temporal_behavior(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Encode each device independently using the shared GRU.

        Parameters
        ----------
        x
            Shape: (B, W, N)

        Returns
        -------
        Tensor
            Temporal device embeddings
            Shape: (B, N, embedding_dim)
        """

        batch_size, window_size, num_devices = x.shape

        #
        # (B,W,N)
        #    ↓
        # (B,N,W)
        #
        x = x.permute(0, 2, 1)

        #
        # (B,N,W)
        #      ↓
        # (B*N,W)
        #
        x = x.reshape(batch_size * num_devices, window_size)

        #
        # GRU expects
        #
        # (B*N,W,1)
        #
        x = x.unsqueeze(-1)

        #
        # Shared GRU
        #
        _, hidden = self.temporal_encoder(x)

        #
        # Last GRU layer
        #
        hidden = hidden[-1]

        #
        # Linear projection
        #
        embeddings = self.embedding_projection(hidden)

        #
        # Restore batch/device layout
        #
        embeddings = embeddings.reshape(
            batch_size,
            num_devices,
            self.embedding_dim,
        )

        return embeddings

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Temporary forward implementation.

        Returns temporal embeddings until graph learning
        and graph reasoning are integrated.
        """

        z = self._encode_temporal_behavior(x)

        return z