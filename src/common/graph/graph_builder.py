from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data


class GraphBuilder:
    """
    Build dynamic interaction graphs from node embeddings.

    The builder is a model-independent graph-construction utility.
    It receives a batch of node embeddings and constructs one directed
    k-nearest-neighbor graph for each sample in the batch.

    The graph topology is computed from cosine similarity between node
    embeddings. The resulting graphs are represented using native
    PyTorch Geometric ``Data`` and ``Batch`` objects.

    This implementation is intended to realize the adaptive graph
    learning stage of HDEG. The builder itself contains no learnable
    parameters and therefore does not inherit from ``torch.nn.Module``.

    Parameters
    ----------
    top_k:
        Number of neighbors selected for every node.

    self_loops:
        If ``False``, a node cannot select itself as a neighbor.
        If ``True``, self-connections are allowed.

    symmetric:
        If ``False``, the graph preserves the directed k-nearest-neighbor
        topology. If ``True``, every selected edge is accompanied by its
        reverse edge.

        Note that symmetric=True can result in more than ``top_k``
        outgoing edges per node because the reverse edges are added
        after neighbor selection.

    Notes
    -----
    Input embeddings are expected to have shape:

        ``(batch_size, num_nodes, embedding_dim)``

    The returned PyG ``Batch`` contains:

        ``x``:
            Flattened node embeddings with shape
            ``(batch_size * num_nodes, embedding_dim)``.

        ``edge_index``:
            Batched graph connectivity with shape ``(2, num_edges)``.

        ``edge_attr``:
            Cosine similarity associated with every edge, with shape
            ``(num_edges, 1)``.

        ``batch``:
            PyG batch-assignment vector identifying the ecosystem to
            which each node belongs.
    """

    def __init__(
        self,
        top_k: int,
        self_loops: bool = False,
        symmetric: bool = False,
    ) -> None:
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not isinstance(self_loops, bool):
            raise TypeError("self_loops must be a boolean.")

        if not isinstance(symmetric, bool):
            raise TypeError("symmetric must be a boolean.")

        self.top_k = top_k
        self.self_loops = self_loops
        self.symmetric = symmetric

    def build(self, embeddings: Tensor) -> Batch:
        """
        Construct a batched dynamic graph from node embeddings.

        Parameters
        ----------
        embeddings:
            Node embeddings with shape
            ``(batch_size, num_nodes, embedding_dim)``.

        Returns
        -------
        torch_geometric.data.Batch
            Batched PyTorch Geometric graph containing one graph for
            every sample in the input batch.

        Raises
        ------
        TypeError
            If ``embeddings`` is not a floating-point tensor.

        ValueError
            If the input tensor does not have rank three, contains
            zero-sized dimensions, contains non-finite values, or
            ``top_k`` is incompatible with the number of nodes.
        """

        self._validate_inputs(embeddings)

        # Normalize node representations so that their dot products
        # correspond to cosine similarities.
        normalized_embeddings = self._normalize_embeddings(
            embeddings
        )

        similarity = self._compute_similarity(
            normalized_embeddings
        )

        neighbor_indices, neighbor_weights = self._select_neighbors(
            similarity
        )

        graphs = []

        batch_size = embeddings.size(0)

        for batch_index in range(batch_size):
            graph = self._build_graph(
                node_embeddings=embeddings[batch_index],
                neighbor_indices=neighbor_indices[batch_index],
                neighbor_weights=neighbor_weights[batch_index],
            )
            graphs.append(graph)

        return Batch.from_data_list(graphs)

    def _validate_inputs(self, embeddings: Tensor) -> None:
        """
        Validate the tensor-level input contract.

        Expected shape:

            ``(batch_size, num_nodes, embedding_dim)``
        """

        if not isinstance(embeddings, Tensor):
            raise TypeError(
                "embeddings must be a torch.Tensor."
            )

        if embeddings.ndim != 3:
            raise ValueError(
                "Expected embeddings with shape "
                "(batch_size, num_nodes, embedding_dim), "
                f"but received shape {tuple(embeddings.shape)}."
            )

        batch_size, num_nodes, embedding_dim = embeddings.shape

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_nodes <= 0:
            raise ValueError(
                "num_nodes must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                "embeddings must be a floating-point tensor."
            )

        if not torch.isfinite(embeddings).all():
            raise ValueError(
                "embeddings contains NaN or infinite values."
            )

        max_neighbors = (
            num_nodes
            if self.self_loops
            else num_nodes - 1
        )

        if self.top_k > max_neighbors:
            if self.self_loops:
                requirement = "top_k <= num_nodes"
            else:
                requirement = "top_k < num_nodes"

            raise ValueError(
                f"Invalid top_k={self.top_k} for "
                f"num_nodes={num_nodes}. "
                f"Expected {requirement}."
            )

    @staticmethod
    def _normalize_embeddings(
        embeddings: Tensor,
    ) -> Tensor:
        """
        L2-normalize every node embedding.

        Parameters
        ----------
        embeddings:
            Tensor of shape ``(B, N, d)``.

        Returns
        -------
        Tensor
            Normalized tensor with the same shape.
        """

        return F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

    @staticmethod
    def _compute_similarity(
        embeddings: Tensor,
    ) -> Tensor:
        """
        Compute pairwise cosine similarity.

        Because the input embeddings have already been L2-normalized,
        cosine similarity reduces to a batched matrix multiplication.

        Parameters
        ----------
        embeddings:
            Normalized embeddings with shape ``(B, N, d)``.

        Returns
        -------
        Tensor
            Pairwise similarity matrix with shape ``(B, N, N)``.
        """

        return torch.matmul(
            embeddings,
            embeddings.transpose(-1, -2),
        )

    def _select_neighbors(
        self,
        similarity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Select the top-k neighbors for every node.

        Parameters
        ----------
        similarity:
            Pairwise similarity matrix with shape ``(B, N, N)``.

        Returns
        -------
        neighbor_indices:
            Tensor with shape ``(B, N, top_k)`` containing the selected
            neighbor indices.

        neighbor_weights:
            Tensor with shape ``(B, N, top_k)`` containing the
            corresponding cosine similarities.
        """

        if similarity.ndim != 3:
            raise ValueError(
                "similarity must have shape (batch_size, num_nodes, "
                "num_nodes)."
            )

        batch_size, num_nodes, _ = similarity.shape

        scores = similarity.clone()

        if not self.self_loops:
            diagonal = torch.eye(
                num_nodes,
                dtype=torch.bool,
                device=scores.device,
            )

            scores = scores.masked_fill(
                diagonal.unsqueeze(0),
                float("-inf"),
            )

        neighbor_weights, neighbor_indices = torch.topk(
            scores,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        if self.symmetric:
            (
                neighbor_indices,
                neighbor_weights,
            ) = self._symmetrize_neighbors(
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                num_nodes=num_nodes,
            )

        return neighbor_indices, neighbor_weights

    @staticmethod
    def _symmetrize_neighbors(
        neighbor_indices: Tensor,
        neighbor_weights: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Add reverse edges to a directed k-nearest-neighbor graph.

        This method preserves the selected edges and appends reverse
        edges. Duplicate edges are removed.

        Parameters
        ----------
        neighbor_indices:
            Tensor with shape ``(B, N, K)``.

        neighbor_weights:
            Tensor with shape ``(B, N, K)``.

        num_nodes:
            Number of nodes in each graph.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Symmetrized neighbor indices and corresponding weights.

        Notes
        -----
        Symmetrization can increase the number of neighbors of a node
        beyond ``K``. Therefore the output has a variable number of
        edges per node after duplicate removal and is converted to
        explicit edge lists during graph construction.
        """

        batch_size = neighbor_indices.size(0)
        top_k = neighbor_indices.size(-1)

        source = (
            torch.arange(
                num_nodes,
                device=neighbor_indices.device,
            )
            .view(1, num_nodes, 1)
            .expand(batch_size, num_nodes, top_k)
        )

        forward_source = source.reshape(batch_size, -1)
        forward_target = neighbor_indices.reshape(batch_size, -1)
        forward_weight = neighbor_weights.reshape(batch_size, -1)

        reverse_source = forward_target
        reverse_target = forward_source
        reverse_weight = forward_weight

        all_source = torch.cat(
            [forward_source, reverse_source],
            dim=-1,
        )

        all_target = torch.cat(
            [forward_target, reverse_target],
            dim=-1,
        )

        all_weight = torch.cat(
            [forward_weight, reverse_weight],
            dim=-1,
        )

        sym_source = []
        sym_target = []
        sym_weight = []

        for batch_index in range(batch_size):
            edges = torch.stack(
                [
                    all_source[batch_index],
                    all_target[batch_index],
                ],
                dim=1,
            )

            unique_edges, inverse = torch.unique(
                edges,
                dim=0,
                return_inverse=True,
            )

            # For duplicated reverse/forward edges, preserve the
            # largest similarity value.
            weights = all_weight[batch_index]

            unique_weights = torch.full(
                (unique_edges.size(0),),
                float("-inf"),
                dtype=weights.dtype,
                device=weights.device,
            )

            unique_weights.scatter_reduce_(
                0,
                inverse,
                weights,
                reduce="amax",
                include_self=True,
            )

            sym_source.append(unique_edges[:, 0])
            sym_target.append(unique_edges[:, 1])
            sym_weight.append(unique_weights)

        max_edges = max(
            edge.size(0)
            for edge in sym_source
        )

        padded_source = []
        padded_target = []
        padded_weight = []

        for source_edges, target_edges, weights in zip(
            sym_source,
            sym_target,
            sym_weight,
        ):
            padding = max_edges - source_edges.size(0)

            if padding > 0:
                source_edges = F.pad(
                    source_edges,
                    (0, padding),
                    value=0,
                )
                target_edges = F.pad(
                    target_edges,
                    (0, padding),
                    value=0,
                )
                weights = F.pad(
                    weights,
                    (0, padding),
                    value=float("-inf"),
                )

            padded_source.append(source_edges)
            padded_target.append(target_edges)
            padded_weight.append(weights)

        return (
            torch.stack(padded_source, dim=0),
            torch.stack(padded_weight, dim=0),
        )

    def _build_graph(
        self,
        node_embeddings: Tensor,
        neighbor_indices: Tensor,
        neighbor_weights: Tensor,
    ) -> Data:
        """
        Construct a PyTorch Geometric Data object for one ecosystem.

        Parameters
        ----------
        node_embeddings:
            Node embeddings with shape ``(N, d)``.

        neighbor_indices:
            Selected neighbor indices.

        neighbor_weights:
            Corresponding edge weights.

        Returns
        -------
        torch_geometric.data.Data
            Graph containing node features, edge connectivity, and
            cosine-similarity edge attributes.
        """

        num_nodes = node_embeddings.size(0)

        if not self.symmetric:
            source_nodes = (
                torch.arange(
                    num_nodes,
                    device=node_embeddings.device,
                )
                .view(-1, 1)
                .expand(-1, self.top_k)
                .reshape(-1)
            )

            target_nodes = neighbor_indices.reshape(-1)
            edge_weights = neighbor_weights.reshape(-1)

        else:
            source_nodes, target_nodes, edge_weights = (
                self._neighbor_lists_to_edges(
                    neighbor_indices,
                    neighbor_weights,
                )
            )

        edge_index = torch.stack(
            [
                source_nodes,
                target_nodes,
            ],
            dim=0,
        ).long()

        edge_attr = edge_weights.unsqueeze(-1)

        return Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

    @staticmethod
    def _neighbor_lists_to_edges(
        neighbor_indices: Tensor,
        neighbor_weights: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert a neighbor-list representation into an edge list.

        Entries whose weight is ``-inf`` are padding entries introduced
        during symmetric graph construction and are discarded.
        """

        source_nodes = (
            torch.arange(
                neighbor_indices.size(0),
                device=neighbor_indices.device,
            )
            .view(-1, 1)
            .expand_as(neighbor_indices)
        )

        valid = torch.isfinite(neighbor_weights)

        source_nodes = source_nodes[valid]
        target_nodes = neighbor_indices[valid]
        edge_weights = neighbor_weights[valid]

        return (
            source_nodes,
            target_nodes,
            edge_weights,
        )