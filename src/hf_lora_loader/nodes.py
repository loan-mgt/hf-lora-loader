from __future__ import annotations

import hashlib
import os
import shutil
from typing import Callable, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

try:
    import folder_paths
except ImportError:  # pragma: no cover - only executed outside ComfyUI
    folder_paths = None

try:
    import nodes
except ImportError:  # pragma: no cover - only executed outside ComfyUI
    nodes = None


HF_SUBDIR = "hf_lora_loader"


class HuggingFaceDownloadError(RuntimeError):
    """Raised when the Hugging Face download fails."""


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def _default_token(token: Optional[str]) -> Optional[str]:
    return token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


def _default_lora_root(lora_root: Optional[str]) -> str:
    if lora_root:
        return lora_root
    if folder_paths is None:
        raise RuntimeError("folder_paths module is required when running inside ComfyUI.")
    roots = folder_paths.get_folder_paths("loras")
    if not roots:
        raise RuntimeError("No 'loras' directory configured in ComfyUI.")
    return roots[0]


def _checksum_matches(path: str, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.lower()


def ensure_hf_lora_file(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
    save_as: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = True,
    expected_sha256: Optional[str] = None,
    lora_root: Optional[str] = None,
    downloader: Optional[Callable[..., str]] = None,
) -> str:
    """Download the requested LoRA file (if necessary) and return its relative path.

    The file is stored under the active ComfyUI `loras` directory inside a dedicated
    sub-folder to avoid clashing with manual downloads. When `expected_sha256` is provided
    the file is verified after download and whenever it already exists locally.
    """

    if not repo_id.strip():
        raise ValueError("A Hugging Face repo_id is required.")
    if not filename.strip():
        raise ValueError("A filename inside the repository is required.")

    downloader = downloader or hf_hub_download
    if downloader is None:
        raise RuntimeError("huggingface_hub is not available. Please install it in your environment.")

    root = _default_lora_root(lora_root)
    repo_slug = _sanitize_repo_id(repo_id.strip())
    final_name = (save_as or filename).strip()
    target_dir = os.path.join(root, HF_SUBDIR, repo_slug)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, final_name)

    expected_sha256 = expected_sha256.lower().strip() if expected_sha256 else ""

    needs_download = force_download or not os.path.exists(target_path)
    if expected_sha256 and os.path.exists(target_path) and not _checksum_matches(target_path, expected_sha256):
        needs_download = True

    if needs_download:
        try:
            download_kwargs = {
                "repo_id": repo_id.strip(),
                "filename": filename.strip(),
                "revision": revision.strip() if revision else None,
                "token": _default_token(token),
                "local_dir": target_dir,
                "local_dir_use_symlinks": False,
                "resume_download": resume_download,
                "force_download": True if force_download else None,
            }
            download_kwargs = {key: value for key, value in download_kwargs.items() if value is not None}
            downloaded_path = downloader(**download_kwargs)
        except HfHubHTTPError as exc:  # pragma: no cover - requires real network access
            raise HuggingFaceDownloadError(f"Failed to download {filename} from {repo_id}: {exc}") from exc

        if os.path.normpath(downloaded_path) != os.path.normpath(target_path):
            shutil.copy2(downloaded_path, target_path)

        if expected_sha256 and not _checksum_matches(target_path, expected_sha256):
            raise ValueError(
                "Downloaded file checksum mismatch. Please verify 'expected_sha256' or disable the check."
            )

    relative_path = os.path.relpath(target_path, root).replace("\\", "/")
    return relative_path


if nodes is not None:

    class HFLoraLoaderModelOnly(nodes.LoraLoaderModelOnly):
        CATEGORY = "loaders/HuggingFace"
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_lora_model_only"
        DESCRIPTION = (
            "Download a LoRA file from Hugging Face if it is missing (or outdated) and "
            "then load it exactly like the built-in LoraLoaderModelOnly node."
        )

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "repo_id": ("STRING", {"default": "author/repo", "tooltip": "owner/repo on huggingface.co"}),
                    "filename": ("STRING", {"default": "model.safetensors", "tooltip": "File path inside the repo."}),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                },
                "optional": {
                    "revision": ("STRING", {"default": "main"}),
                    "save_as": ("STRING", {"default": "", "tooltip": "Override the local filename (optional)."}),
                    "expected_sha256": ("STRING", {"default": "", "tooltip": "Optional checksum to enforce integrity."}),
                    "huggingface_token": ("STRING", {"default": "", "tooltip": "Overrides HF_TOKEN env."}),
                    "force_download": ("BOOLEAN", {"default": False}),
                    "resume_download": ("BOOLEAN", {"default": True}),
                },
            }

        def load_lora_model_only(
            self,
            model,
            repo_id,
            filename,
            strength_model,
            revision="main",
            save_as="",
            expected_sha256="",
            huggingface_token="",
            force_download=False,
            resume_download=True,
        ):  # pylint: disable=arguments-differ
            local_name = ensure_hf_lora_file(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                save_as=save_as or None,
                token=huggingface_token or None,
                force_download=bool(force_download),
                resume_download=bool(resume_download),
                expected_sha256=expected_sha256 or None,
            )
            return super().load_lora_model_only(model, local_name, strength_model)


    NODE_CLASS_MAPPINGS = {
        "HFLoraLoaderModelOnly": HFLoraLoaderModelOnly,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "HFLoraLoaderModelOnly": "HF LoRA Loader (Model Only)",
    }

else:  # pragma: no cover - executed only in environments without ComfyUI
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
