"""Plot payload builders for clustering diagnostics.

Each module adds a small, focused section to the overall `plot_data` payload.
The public entrypoint is orchestrated by `engine.reporting.diagnostics.clustering.extras`.
"""

from .cluster_sizes import add_cluster_sizes
from .decoder import add_decoder_payload
from .dendrogram import add_dendrogram
from .distance_to_center import add_distance_to_center
from .elbow import add_elbow_curve
from .embedding import add_embedding_labels
from .gmm_ellipses import add_gmm_ellipses
from .k_distance import add_k_distance_and_dbscan_counts
from .profiles import add_centroid_profiles
from .separation import add_separation
from .silhouette import add_silhouette
from .spectral import add_spectral_eigenvalues

__all__ = [
    "add_cluster_sizes",
    "add_distance_to_center",
    "add_centroid_profiles",
    "add_separation",
    "add_silhouette",
    "add_decoder_payload",
    "add_elbow_curve",
    "add_k_distance_and_dbscan_counts",
    "add_spectral_eigenvalues",
    "add_dendrogram",
    "add_embedding_labels",
    "add_gmm_ellipses",
]
