"""Run the reviewed SPY42 notebook trainer on a standard GitHub CPU runner.

The trained artifact is compatible with the existing Colab notebook. No personal
token, browser session, new data, or final-test evaluation is required.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

REPO = "jamelof23/SPY"
DATASET_TAG = "spy42-data-2026-11-22-v1"
MODEL_TAG = "spy42-model-v1"
EXPECTED_FILES = {
    "github_artifacts.py": "f6151d10b5d75283a08fac3ba013e92cc1df5894de20f3f1efda90c851714618",
    "spy42_workflow.py": "e906b5f288e907c66be7e1690e88427811e0e684263de5d3ff1e9064aa3c67d4",
    "train_spy_images.py": "4d99920b6bc2402f114cfc4ba9dee26d815904c146b5f6436c7d4aed17812979",
    "build_spy_dataset.py": "8f0a572d0b775ac99fa9fb4500b78fa8202bd16f2a58206c5c30f629c11d74c0",
    "SPY_daily_with_2019_warmup.csv": "4e2045b50251b87619b406abcdee3697862a2d69cc18744075fc15683ab57c5d",
}


def summary(message):
    print(message, flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def unpack_bundle(notebook_path, target_dir):
    """Read literal bundled data only; never execute notebook cells."""
    notebook_path, target_dir = Path(notebook_path), Path(target_dir)
    if not notebook_path.is_file() or notebook_path.stat().st_size > 2 * 1024**2:
        raise ValueError("The reviewed notebook is missing or oversized.")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    payloads = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        if "BUNDLED_FILES" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "BUNDLED_FILES"
                for target in node.targets
            ):
                payloads.append(ast.literal_eval(node.value))
    if len(payloads) != 1 or not isinstance(payloads[0], dict):
        raise ValueError("Expected one literal notebook bundle.")
    payload = payloads[0]
    if set(payload) != set(EXPECTED_FILES):
        raise ValueError("Bundled file set differs from the reviewed training code.")
    verified = {}
    for name, expected in EXPECTED_FILES.items():
        packed = payload[name]
        if not isinstance(packed, dict) or packed.get("sha256") != expected:
            raise ValueError("Bundled code version differs: " + name)
        decoder = zlib.decompressobj()
        content = decoder.decompress(
            base64.b64decode(packed["zlib_base64"], validate=True), 2 * 1024**2 + 1
        )
        if (len(content) > 2 * 1024**2 or decoder.unconsumed_tail
                or decoder.unused_data or not decoder.eof
                or hashlib.sha256(content).hexdigest() != expected):
            raise ValueError("Bundled file failed verification: " + name)
        verified[name] = content
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, content in verified.items():
        (target_dir / name).write_bytes(content)
    return target_dir


def actions_client(base_class, error_class, token):
    class ActionsGitHubArtifacts(base_class):
        """Use the same installation-token preflight as the dataset publisher."""

        def require_write_access(self):
            if (os.environ.get("GITHUB_ACTIONS") != "true"
                    or os.environ.get("GITHUB_REPOSITORY", "").lower() != self.repo.lower()
                    or not self._token):
                raise error_class("Expected this repository's explicit Actions token.")
            info = self._api("GET", "", authenticated=True)
            if (not isinstance(info, dict)
                    or str(info.get("full_name", "")).lower() != self.repo.lower()
                    or info.get("private") is not False):
                raise error_class("Expected the authorized public SPY repository.")
            # A repository-role flag does not establish an installation token's
            # app permissions. The release write enforces Contents:write; actual
            # authorization errors propagate without an alternative credential.
            return info

    return ActionsGitHubArtifacts(REPO, token=token)


def verify_existing_model(github, workflow, release, work_root):
    """Skip a compatible published model; incomplete versions never retrain."""
    if release is None:
        return False
    if release.get("draft") or not workflow.release_has_assets(
        release, [workflow.MODEL_ASSET, workflow.MODEL_METADATA]
    ):
        raise RuntimeError("An incomplete model release exists; recover its original run before retrying.")
    path = Path(work_root) / "existing_model_metadata.json"
    github.download_asset(release, workflow.MODEL_METADATA, path)
    meta = workflow.read_json(path)
    workflow._check_metadata(meta, "spy42_model", REPO, MODEL_TAG)
    expected = {
        "trainer_sha256": EXPECTED_FILES["train_spy_images.py"],
        "dataset_tag_at_training": DATASET_TAG,
        "archive_name": workflow.MODEL_ASSET,
        "smoke_only": False,
        "test_target_values_loaded_during_training": False,
        "training_years": [2020, 2021, 2022, 2023, 2024],
        "validation_years": [2025],
    }
    if any(meta.get(key) != value for key, value in expected.items()) or not meta.get("completed_at_utc"):
        raise RuntimeError("The existing model uses a different or incomplete training contract.")
    assets = {a["name"]: a for a in release["assets"]}
    archive = assets[workflow.MODEL_ASSET]
    if archive.get("size", 0) <= 0 or (
        archive.get("digest") is not None
        and archive["digest"] != "sha256:" + meta["archive_sha256"]
    ):
        raise RuntimeError("The published model asset metadata disagree.")
    summary("A completed compatible model already exists. Training and publication skipped.")
    summary(f"[Open saved model](https://github.com/{REPO}/releases/tag/{MODEL_TAG})")
    return True


def training_commands(python, trainer_path, data_root, run_dir):
    base = [str(python), "-u", str(trainer_path), "--data-root", str(data_root),
            "--output-dir", str(run_dir)]
    smoke = base + ["--smoke-test", "--workers", "0"]
    full = base + [
        "--device", "cpu", "--epochs", "15", "--batch-size", "16", "--workers", "2",
        "--pretrained", "--height", "336", "--width", "768",
        "--learning-rate", "0.001", "--quantile-loss-weight", "0.25",
        "--patience", "4", "--seed", "42", "--min-validation-signals", "30",
        "--proxy-cost-bps", "0.0",
    ]
    return smoke, full


def run_logged(command, log_path):
    """Keep nonsensitive training output; the trainer does not receive the token."""
    child_env = os.environ.copy()
    child_env.pop("GITHUB_TOKEN", None)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=child_env)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Trainer exited with status {code}; saved outputs were preserved.")


def require_completed_contract(workflow, run_dir):
    if not workflow.is_completed_run(run_dir):
        raise RuntimeError("Full training did not complete; no model will be published.")
    config = workflow.read_json(Path(run_dir) / "run_config.json")
    expected = {
        "smoke_only": False, "test_target_values_loaded": False,
        "input_features": "image_pixels_only", "device": "cpu",
        "training_years": [2020, 2021, 2022, 2023, 2024], "validation_years": [2025],
        "train_image_count": 1258, "validation_image_count": 250,
        "transform": {"height": 336, "width": 768},
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Completed training does not match the reviewed experiment.")
    architecture = config.get("architecture", {})
    if architecture.get("training_strategy") != "frozen" or architecture.get("pretrained_initialization") is not True:
        raise RuntimeError("Unexpected model initialization or training strategy.")
    return config


def main():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != REPO
            or os.environ.get("GITHUB_REF") != "refs/heads/main"):
        raise RuntimeError("Run this workflow only on jamelof23/SPY main.")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub did not provide the repository job token.")
    work_root = Path(os.environ["RUNNER_TEMP"]) / "spy42_model_training"
    bootstrap_dir = unpack_bundle("SPY_42_Image_Training_Colab.ipynb", work_root / "runtime_code")
    sys.path.insert(0, str(bootstrap_dir))
    from github_artifacts import GitHubArtifacts, GitHubArtifactError, file_sha256
    import spy42_workflow as workflow

    github = actions_client(GitHubArtifacts, GitHubArtifactError, token)
    github.require_write_access()
    if verify_existing_model(github, workflow, github.get_release(MODEL_TAG), work_root):
        return
    data_root = workflow.ensure_dataset(
        github, DATASET_TAG, work_root, bootstrap_dir, allow_bootstrap=False, workers=2
    )
    fingerprint = workflow.training_fingerprint(data_root)
    trainer_path = bootstrap_dir / "train_spy_images.py"
    trainer_sha = file_sha256(trainer_path)
    run_dir = work_root / "local_runs" / MODEL_TAG
    context_path = work_root / "local_run_contexts" / (MODEL_TAG + ".json")
    context_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_json(context_path, {
        "repo": REPO, "model_tag": MODEL_TAG,
        "training_data_fingerprint": fingerprint, "trainer_sha256": trainer_sha,
    })
    summary("## SPY42 training on GitHub")
    summary("42 daily candles, EMA8, EMA21, and volume; model input is chart pixels only.")
    summary("Training: 2020–2024. Validation: 2025. Final-test evaluation: disabled.")
    smoke, full = training_commands(sys.executable, trainer_path, data_root, run_dir)
    summary("Running the separate 4+4-image wiring check before full training.")
    run_logged(smoke, work_root / "logs" / "smoke.log")
    summary("Starting full training: 1,258 training images and 250 validation images.")
    run_logged(full, work_root / "logs" / "training.log")
    config = require_completed_contract(workflow, run_dir)
    workflow.write_json(run_dir / "execution_context.json", {
        "execution_platform": "GitHub Actions standard Ubuntu CPU runner",
        "repository": REPO, "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_script_sha256": file_sha256(Path(__file__)),
        "trainer_sha256": trainer_sha, "training_data_fingerprint": fingerprint,
        "python": sys.version, "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "numpy", "pandas", "Pillow", "requests"]
        },
        "training_command": full, "final_test_evaluation": False,
    })
    published = workflow.publish_model(
        github, MODEL_TAG, run_dir, work_root, fingerprint, trainer_sha, DATASET_TAG
    )
    # Verify the authenticated release ID, avoiding stale anonymous tag caches
    # immediately after publication. Asset digests are compared with local files.
    release_id = published.get("id")
    if isinstance(release_id, bool) or not isinstance(release_id, int) or release_id <= 0:
        raise RuntimeError("Publication did not return a valid release ID.")
    confirmed = github._api("GET", f"/releases/{release_id}", authenticated=True)
    if confirmed.get("tag_name") != MODEL_TAG or confirmed.get("draft"):
        raise RuntimeError("The model release was not confirmed as published.")
    artifact_dir = work_root / "model_artifacts" / MODEL_TAG
    expected_assets = {
        name: {"size": (artifact_dir / name).stat().st_size,
               "sha256": file_sha256(artifact_dir / name)}
        for name in [workflow.MODEL_ASSET, workflow.MODEL_METADATA]
    }
    github._check_assets(confirmed, expected_assets, require_digests=False, complete=True)
    summary(f"Completed full training; selected epoch {config['best_epoch']} using 2025 validation.")
    summary(f"[Download saved model](https://github.com/{REPO}/releases/tag/{MODEL_TAG})")
    summary("The existing Colab notebook can now load this model in AUTO mode without a personal token.")
    summary("Validation results are development measurements; no 2026 final-test outcomes were evaluated.")


if __name__ == "__main__":
    main()
