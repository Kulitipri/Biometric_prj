"""
Embedding Extraction for Inference.

Module này là cầu nối giữa model đã train và các tác vụ xuôi dòng:
- Trainer: extract embeddings cho validation, gọi metrics
- evaluate.py: extract embeddings cho test pairs
- Demo/deploy: real-time verification, identification

Các thành phần:
- EmbeddingExtractor: class wrapping batch extraction
- verify(): 1-vs-1 verification từ 2 embeddings
- identify(): 1-vs-N identification trong gallery
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ====================================================================== #
# EmbeddingExtractor
# ====================================================================== #


class EmbeddingExtractor:
    """
    Wrapper cho batch extraction từ FaceRecognizer hoặc bất kỳ embedding model.

    Tự động:
    - Set model về eval mode (disable dropout, BN frozen)
    - Move model sang device
    - Disable gradient computation
    - Return numpy arrays (L2-normalized) để dễ xử lý xuôi dòng

    Args:
        model: nn.Module có method extract_embedding(x) hoặc forward(x).
            Khuyến nghị FaceRecognizer (có sẵn extract_embedding với L2 norm).
        device: 'cuda' / 'cpu' / 'cuda:0'.
            Tự fallback về CPU nếu CUDA không khả dụng.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        # Auto fallback CPU nếu CUDA không có
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng, fallback về CPU")
            device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

        # Cache method extract_embedding nếu có (cho FaceRecognizer)
        # Nếu không có thì fallback về forward
        self._has_extract_method = hasattr(model, "extract_embedding")

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings cho 1 batch.

        Đây là method "thấp nhất" - chỉ xử lý 1 batch.
        Các method khác (extract_from_loader, extract_pairs) gọi method này.

        Args:
            images: (B, 3, 112, 112) tensor.
                Có thể trên CPU hoặc GPU - sẽ tự move sang self.device.

        Returns:
            embeddings: (B, embedding_dim) numpy array, L2-normalized
                (nếu model có extract_embedding method).
        """
        if images.dim() != 4:
            raise ValueError(
                f"Images phải 4D (B, C, H, W), nhận shape {tuple(images.shape)}"
            )

        # Move tensor sang đúng device
        images = images.to(self.device, non_blocking=True)

        # Forward pass
        if self._has_extract_method:
            # FaceRecognizer.extract_embedding đã L2-normalize
            embeddings = self.model.extract_embedding(images)
        else:
            # Generic model: gọi forward và tự L2 normalize
            output = self.model(images)
            # Một số model trả về tuple (embedding, logits)
            if isinstance(output, tuple):
                output = output[0]
            embeddings = torch.nn.functional.normalize(output, p=2, dim=1)

        # Convert sang numpy. Cần .cpu() trước khi numpy()
        return embeddings.cpu().numpy()

    def extract_from_loader(
        self,
        loader: DataLoader,
        show_progress: bool = True,
        desc: str = "Extracting embeddings",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings cho cả DataLoader (typical FaceDataset).

        Loader phải yield (images, labels) - format của FaceDataset.

        Args:
            loader: DataLoader yield (images, labels).
            show_progress: True = hiển thị tqdm.
            desc: Description cho progress bar.

        Returns:
            (all_embeddings, all_labels):
                - all_embeddings: (N, D) numpy.
                - all_labels: (N,) numpy int.
        """
        all_embeddings: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator = tqdm(loader, desc=desc) if show_progress else loader

        for batch in iterator:
            # Hỗ trợ cả 2 format: (images, labels) hoặc chỉ images
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
                # Convert labels sang numpy ngay (có thể là tensor)
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                all_labels.append(np.asarray(labels))
            else:
                images = batch

            embeddings = self.extract(images)
            all_embeddings.append(embeddings)

        # Concat tất cả batches
        embeddings_arr = np.concatenate(all_embeddings, axis=0)
        labels_arr = (
            np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        )

        logger.info(
            f"Extracted {embeddings_arr.shape[0]} embeddings, "
            f"dim={embeddings_arr.shape[1]}"
        )

        return embeddings_arr, labels_arr

    def extract_pairs(
        self,
        loader: DataLoader,
        show_progress: bool = True,
        desc: str = "Extracting pair embeddings",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings cho VerificationDataset (yield (img1, img2, label)).

        Hiệu quả hơn so với gọi extract_from_loader 2 lần vì chỉ loop 1 lần
        qua data và xử lý 2 ảnh trong cùng batch.

        Args:
            loader: DataLoader yield (img1, img2, label).
            show_progress: True = hiển thị tqdm.
            desc: Description cho progress bar.

        Returns:
            (embeddings1, embeddings2, labels):
                - embeddings1: (N, D) embeddings của ảnh 1.
                - embeddings2: (N, D) embeddings của ảnh 2.
                - labels: (N,) 1=same, 0=different.
        """
        all_emb1: List[np.ndarray] = []
        all_emb2: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator = tqdm(loader, desc=desc) if show_progress else loader

        for batch in iterator:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                raise ValueError(
                    f"VerificationDataset phải yield (img1, img2, label), "
                    f"nhận: {type(batch)}"
                )
            img1, img2, label = batch

            # Trick: ghép 2 ảnh thành 1 batch lớn rồi extract 1 lần
            # → tận dụng GPU parallelism
            combined = torch.cat([img1, img2], dim=0)  # (2B, 3, 112, 112)
            embeddings = self.extract(combined)         # (2B, D)

            # Split ngược lại
            batch_size = img1.shape[0]
            emb1 = embeddings[:batch_size]
            emb2 = embeddings[batch_size:]

            all_emb1.append(emb1)
            all_emb2.append(emb2)

            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
            all_labels.append(np.asarray(label))

        emb1_arr = np.concatenate(all_emb1, axis=0)
        emb2_arr = np.concatenate(all_emb2, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        logger.info(
            f"Extracted {emb1_arr.shape[0]} pairs, embedding_dim={emb1_arr.shape[1]}"
        )

        return emb1_arr, emb2_arr, labels_arr


# ====================================================================== #
# Standalone functions: verify and identify
# ====================================================================== #


def verify(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Verification: 2 ảnh có phải cùng người không?

    Args:
        embedding1: (D,) L2-normalized embedding.
        embedding2: (D,) L2-normalized embedding.
        threshold: Cosine similarity threshold để quyết định cùng người.
            Default 0.5 - nên thay bằng giá trị từ validation set.

    Returns:
        (is_same_person, similarity):
            - is_same_person: True nếu similarity >= threshold.
            - similarity: cosine similarity, giá trị trong [-1, 1].

    Examples:
        >>> emb1 = extractor.extract(image1[None])[0]  # (D,)
        >>> emb2 = extractor.extract(image2[None])[0]
        >>> is_same, score = verify(emb1, emb2, threshold=0.4)
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Shape mismatch: {embedding1.shape} vs {embedding2.shape}"
        )

    # Với L2-normalized vectors: cosine = dot product
    similarity = float(np.dot(embedding1, embedding2))
    is_same = bool(similarity >= threshold)
    return is_same, similarity


def identify(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: List,
    top_k: int = 5,
    threshold: float = None,
) -> List[Tuple[object, float]]:
    """
    Identification: query ảnh giống nhất với ai trong gallery?

    1-to-N matching task: cho 1 query, tìm top_k matches trong gallery N người.

    Args:
        query_embedding: (D,) embedding của ảnh query (L2-normalized).
        gallery_embeddings: (N, D) embeddings của gallery (L2-normalized).
        gallery_labels: List[N] identities/names tương ứng.
        top_k: Số kết quả top trả về.
        threshold: Nếu set, chỉ trả về matches có similarity >= threshold.
            None = trả về top_k bất kể similarity.

    Returns:
        List[(label, similarity)] sorted desc theo similarity.
        Có thể có ít hơn top_k items nếu có threshold filter.

    Examples:
        >>> # Build gallery (1 lần)
        >>> gallery_embs = extractor.extract_from_loader(gallery_loader)[0]
        >>> gallery_labels = ['Alice', 'Bob', 'Charlie', ...]
        >>>
        >>> # Identify query
        >>> query_emb = extractor.extract(query_image[None])[0]
        >>> matches = identify(query_emb, gallery_embs, gallery_labels, top_k=3)
        >>> # [('Alice', 0.85), ('Bob', 0.42), ('Charlie', 0.31)]
    """
    if gallery_embeddings.ndim != 2:
        raise ValueError(
            f"gallery_embeddings phải 2D (N, D), nhận {gallery_embeddings.shape}"
        )
    if len(gallery_labels) != gallery_embeddings.shape[0]:
        raise ValueError(
            f"Số labels ({len(gallery_labels)}) khác số embeddings "
            f"({gallery_embeddings.shape[0]})"
        )
    if query_embedding.shape[0] != gallery_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: query={query_embedding.shape[0]}, "
            f"gallery={gallery_embeddings.shape[1]}"
        )

    # Tính similarity query vs tất cả gallery
    # Vì cả 2 đều L2-normalized: cosine = matrix-vector product
    similarities = gallery_embeddings @ query_embedding  # (N,)

    # Lấy top_k cao nhất
    # argsort default ascending → đảo ngược để lấy desc
    # Dùng argpartition cho hiệu quả khi N >> top_k
    if top_k >= len(similarities):
        # Lấy hết → sort thường
        top_indices = np.argsort(-similarities)
    else:
        # Lấy top_k qua argpartition (O(N) thay vì O(N log N))
        # rồi sort lại trong top_k đó
        partition_idx = np.argpartition(-similarities, top_k)[:top_k]
        # Sort top_k này theo similarity desc
        top_indices = partition_idx[np.argsort(-similarities[partition_idx])]

    # Build results, optional filter theo threshold
    results: List[Tuple[object, float]] = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if threshold is not None and sim < threshold:
            break  # Vì đã sort desc, gặp 1 cái dưới threshold → dừng
        results.append((gallery_labels[idx], sim))

    return results