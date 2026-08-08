from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.common.graph.graph_builder import GraphBuilder
from src.models.hdeg.graph_reasoner import GraphReasoner


class DBRL(nn.Module):
    """
    Device Behavioral Representation Learning (DBRL).

    DBRL transforms a multivariate smart-home observation window into
    contextualized device-level behavioral representations.

    Computational pipeline
    ----------------------
    1. Tensor organization
    2. Shared temporal behavior encoding using a GRU
    3. Adaptive graph learning using GraphBuilder
    4. Graph relational reasoning using GraphReasoner

    Input
    -----
    x:
        Tensor of shape ``(B, W, N)`` where:

        B = batch size
        W = temporal window length
        N = number of devices

    Output
    ------
    Tensor of shape ``(B, N, D)`` where:

        D = behavioral representation dimensionality

    Scientific artifact
    -------------------
    The returned tensor is the contextualized device behavioral
    representation set Z.

    Intermediate temporal embeddings, the learned interaction graph,
    and graph message representations remain internal to the module.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.0,
        graph_top_k: int = 15,
        graph_self_loops: bool = False,
        graph_symmetric: bool = False,
        graph_heads: int = 1,
        graph_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gru_layers = gru_layers

        # ---------------------------------------------------------
        # Temporal Behavior Encoder
        # ---------------------------------------------------------
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ---------------------------------------------------------
        # Temporal Embedding Projection
        # ---------------------------------------------------------
        self.embedding_projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

        # ---------------------------------------------------------
        # Adaptive Graph Learning
        # ---------------------------------------------------------
        self.graph_builder = GraphBuilder(
            top_k=graph_top_k,
            self_loops=graph_self_loops,
            symmetric=graph_symmetric,
        )

        # ---------------------------------------------------------
        # Graph Relational Reasoning
        # ---------------------------------------------------------
        self.graph_reasoner = GraphReasoner(
            embedding_dim=embedding_dim,
            heads=graph_heads,
            dropout=graph_dropout,
        )

    def _encode_temporal_behavior(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Encode each device independently using the shared GRU.

        Parameters
        ----------
        x:
            Input observation tensor with shape
            ``(B, W, N)``.

        Returns
        -------
        Tensor
            Temporal device embeddings with shape
            ``(B, N, embedding_dim)``.
        """

        batch_size, window_size, num_devices = x.shape

        # ---------------------------------------------------------
        # (B, W, N)
        #     ↓
        # (B, N, W)
        #
        # Each device now owns one temporal sequence.
        # ---------------------------------------------------------
        x = x.permute(0, 2, 1)

        # ---------------------------------------------------------
        # (B, N, W)
        #     ↓
        # (B*N, W)
        # ---------------------------------------------------------
        x = x.reshape(
            batch_size * num_devices,
            window_size,
        )

        # ---------------------------------------------------------
        # GRU input:
        #
        # (B*N, W)
        #     ↓
        # (B*N, W, 1)
        # ---------------------------------------------------------
        x = x.unsqueeze(-1)

        # ---------------------------------------------------------
        # Shared temporal encoder
        # ---------------------------------------------------------
        _, hidden = self.temporal_encoder(x)

        # Last layer of the GRU.
        #
        # hidden shape:
        # (gru_layers, B*N, hidden_dim)
        #
        # ↓
        #
        # (B*N, hidden_dim)
        # ---------------------------------------------------------
        hidden = hidden[-1]

        # ---------------------------------------------------------
        # Project into the DBRL representation space.
        # ---------------------------------------------------------
        embeddings = self.embedding_projection(
            hidden
        )

        # ---------------------------------------------------------
        # Restore:
        #
        # (B*N, embedding_dim)
        #     ↓
        # (B, N, embedding_dim)
        # ---------------------------------------------------------
        embeddings = embeddings.reshape(
            batch_size,
            num_devices,
            self.embedding_dim,
        )

        return embeddings

    @staticmethod
    def _validate_input(
        x: Tensor,
    ) -> None:
        """
        Validate the DBRL input tensor.
        """

        if not isinstance(x, Tensor):
            raise TypeError(
                "DBRL input must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "DBRL expects input with shape "
                "(batch_size, window_size, num_devices). "
                f"Received shape {tuple(x.shape)}."
            )

        batch_size, window_size, num_devices = x.shape

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if num_devices <= 0:
            raise ValueError(
                "num_devices must be greater than zero."
            )

        if not x.is_floating_point():
            raise TypeError(
                "DBRL input must be a floating-point tensor."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "DBRL input contains NaN or infinite values."
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Execute the complete DBRL computational pipeline.

        Parameters
        ----------
        x:
            Observation window with shape
            ``(B, W, N)``.

        Returns
        -------
        Tensor
            Contextualized device behavioral representations with
            shape ``(B, N, embedding_dim)``.
        """

        self._validate_input(x)

        # =========================================================
        # Step 1 + Step 2
        #
        # Temporal behavior encoding
        # =========================================================
        temporal_embeddings = (
            self._encode_temporal_behavior(x)
        )

        # =========================================================
        # Step 3
        #
        # Adaptive graph learning
        #
        # The graph is explicitly constructed from the temporal
        # embeddings, not from the raw observations.
        # =========================================================
        interaction_graph = (
            self.graph_builder.build(
                temporal_embeddings
            )
        )

        # =========================================================
        # Step 4
        #
        # Graph relational reasoning
        # =========================================================
        behavioral_representations = (
            self.graph_reasoner(
                interaction_graph
            )
        )

        # =========================================================
        # Final representation validation
        # =========================================================
        if not torch.isfinite(
            behavioral_representations
        ).all():
            raise RuntimeError(
                "DBRL produced NaN or infinite behavioral "
                "representations."
            )

        return behavioral_representations