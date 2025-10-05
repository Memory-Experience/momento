from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


class HuggingFaceHelper:
    """
    Helper utility for HuggingFace model operations.

    Handles automatic downloading and resolution of GGUF model files from
    HuggingFace repositories, with intelligent quantization selection based
    on preferences. Supports dependency injection for testing.
    """

    DEFAULT_PREFERRED_QUANTS: Sequence[str] = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0")

    def __init__(
        self,
        *,
        hf_hub_download_fn: Callable | None = None,
        list_repo_files_fn: Callable | None = None,
    ):
        """
        Initialize with optional dependency injection for testing.

        Args:
            hf_hub_download_fn: Function to download files from HF,
                defaults to huggingface_hub.hf_hub_download
            list_repo_files_fn: Function to list repo files,
                defaults to huggingface_hub.list_repo_files
        """
        self._hf_hub_download = hf_hub_download_fn or hf_hub_download
        self._list_repo_files = list_repo_files_fn or list_repo_files

    def ensure_local_model(
        self,
        *,
        model_path: str | None = None,
        hf_repo_id: str | None = None,
        download_dir: str = "models/llm",
        preferred_quants: Sequence[str] = DEFAULT_PREFERRED_QUANTS,
    ) -> str:
        """
        Ensure a GGUF model is available locally.

        If a local path is provided, validates it exists. Otherwise,
        downloads the model from HuggingFace, intelligently selecting
        the best available quantization based on the preferred_quants
        list.

        Parameters:
            model_path: Path to a local model file (optional)
            hf_repo_id: HuggingFace repository ID to download from
                (required if no model_path)
            download_dir: Directory to download models to
                (default: "models/llm")
            preferred_quants: Quantization levels to prefer, in order of
                preference (e.g., ["Q4_K_M", "Q4_K_S", "Q5_K_M"])

        Returns:
            Absolute path to the local model file

        Raises:
            FileNotFoundError: If model_path doesn't exist or no GGUF
                files found in repo
            ValueError: If neither model_path nor hf_repo_id is provided
        """
        if model_path:
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            return str(p)

        if not hf_repo_id:
            raise ValueError(
                "Provide either a local `model_path` or an HF `hf_repo_id` to download."
            )

        # Get current script directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_download_dir = os.path.join(script_dir, download_dir)

        import logging

        logging.info(f"Using download directory: {abs_download_dir}")

        # List files in the repo
        files = self._list_repo_files(hf_repo_id)
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found in repo {hf_repo_id}")

        # Pick the best quantization based on preferences
        def pick() -> str:
            for q in preferred_quants:
                cand = [f for f in ggufs if re.search(rf"\b{re.escape(q)}\b", f)]
                if cand:
                    return max(cand, key=len)
            return max(ggufs, key=len)

        chosen = pick()

        # Create directory if it doesn't exist
        os.makedirs(abs_download_dir, exist_ok=True)

        # Download the model
        path = self._hf_hub_download(
            repo_id=hf_repo_id,
            filename=chosen,
            local_dir=abs_download_dir,
        )

        # Return the absolute path to the downloaded file
        return str(Path(path).resolve())
