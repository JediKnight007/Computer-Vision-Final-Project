import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def _euclidean_distance(a: float, b: float) -> float:
    """Euclidean distance between two scalar grayscale pixel values."""
    return abs(a - b)

def _cosine_similarity(a: float, b: float) -> float:
    """Cosine similarity between two colour pixel values."""
    if a == 0 and b == 0:
        return 1.0  # Both pixels are black, consider them identical
    return (a * b) / (np.sqrt(a**2) * np.sqrt(b**2))



def graph_to_image(G: nx.Graph, H: int = 28, W: int = 28) -> np.ndarray:
    """
    Reconstruct a (H, W) float32 image from a graph produced by image_to_graph.
    Uses the 'row', 'col', and 'value' node attributes.
    """
    image = np.zeros((H, W), dtype=np.float32)
    for _, data in G.nodes(data=True):
        image[data["row"], data["col"]] = data["value"]
    return image

class InitialSegmenter:

    def __init__(self, pixel_dist_function: str = "euclidean", cutoff: float = 0.1):
        
        self.cutoff = cutoff
        
        match pixel_dist_function:
            case "euclidean":
                self._dist_func = np.linalg.norm
            case "cosine":
                self._dist_func = np.cosine_similarity
            case _:
                raise ValueError(f"Unsupported pixel distance function: {pixel_dist_function}")


    def image_to_graph(self, image: np.ndarray) -> nx.Graph:
        """
        Convert a (H, W) grayscale or colour image (values in [0, 1]) to a NetworkX graph.

        Each pixel becomes a node with:
          - 'value': normalized pixel intensity (float)
          - 'row', 'col': position in the image

        Adjacent pixels (4-connectivity: up/down/left/right) are connected by edges.

        Returns:
            G: nx.Graph with H*W nodes and ~2*H*W - H - W edges
        """

        if image.ndim == 2:
            H, W = image.shape
        elif image.ndim == 3 and image.shape[2] == 1:
            H, W = image.shape[:2]
            image = image.squeeze(axis=2)  # Convert to (H, W)
        elif image.ndim == 3 and image.shape[2] == 3:
            H, W, _ = image.shape

        G = nx.Graph()

        # Add all nodes
        for r in range(H):
            for c in range(W):
                node_id = r * W + c
                G.add_node(node_id, value=float(image[r, c]), row=r, col=c)

        # Add edges for 4-connected neighbours with cosine similarity weights
        for r in range(H):
            for c in range(W):
                node_id = r * W + c
                val = G.nodes[node_id]["value"]
                if c + 1 < W:                          # right neighbour
                    right_id = node_id + 1
                    weight = self._dist_func(val, G.nodes[right_id]["value"])
                    G.add_edge(node_id, right_id, weight=weight)
                if r + 1 < H:                          # bottom neighbour
                    below_id = node_id + W
                    weight = self._dist_func(val, G.nodes[below_id]["value"])
                    G.add_edge(node_id, below_id, weight=weight)

        return G
    
    def segment_graph(self, G: nx.Graph, attribute: str = 'ricciCurvature') -> nx.Graph:
        """
        Segment the graph by removing edges with Ricci curvature above the tolerance threshold.

        Args:
            G: Input graph with attributes on edges.

        Returns:
            G_segmented: Graph with edges removed, potentially resulting in disconnected components.
        """
        G_segmented = G.copy()
        for u, v, data in G.edges(data=True):
            if data[attribute] > self.cutoff:
                G_segmented.remove_edge(u, v)
        return G_segmented

    def graph_to_segmentation_map(self, G: nx.Graph, H: int = 28, W: int = 28) -> np.ndarray:
        """
        Convert a (possibly disconnected) pixel graph into a 2D segmentation map.

        Each connected component is assigned a unique integer cluster ID (0-indexed,
        in order of first discovery). The returned array has shape (H, W) where
        entry [r, c] is the cluster ID of the pixel at row r, column c.

        Args:
            G:    Graph produced by image_to_graph, possibly after operations that
                  disconnect nodes into clusters.
            H, W: Image dimensions (default 28x28).

        Returns:
            seg: (H, W) int32 array of cluster IDs.
        """
        seg = np.full((H, W), -1, dtype=np.int32)
        for cluster_id, component in enumerate(nx.connected_components(G)):
            for node in component:
                data = G.nodes[node]
                seg[data["row"], data["col"]] = cluster_id
        return seg


    def segment(self, image: np.ndarray) -> np.ndarray:
        
        if image.ndim == 2:
            H, W = image.shape
        elif image.ndim == 3 and image.shape[2] == 1:
            H, W = image.shape[:2]
            image = image.squeeze(axis=2)  # Convert to (H, W)
        elif image.ndim == 3 and image.shape[2] == 3:
            H, W, _ = image.shape
        
        graph = self.image_to_graph(image)
        orc_runner = OllivierRicci(graph, verbose="INFO")
        orc_runner.compute_ricci_flow(iterations=15)
        segmented_map = self.graph_to_segmentation_map(orc_runner.G, H, W)

        return segmented_map