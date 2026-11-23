"""Install isolated benchmark dependencies and pinned upstream evaluation assets."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
import venv
from zipfile import ZipFile


API_BANK_ENCODER = "sentence-transformers/paraphrase-MiniLM-L3-v2"
API_BANK_ENCODER_REV = "4ca70771034acceecb2e72475f72050fcdde4ddc"
API_BANK_ENCODER_SHA256 = (
    "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
)
MEMORY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_ENCODER_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def pip(worker, *arguments):
    subprocess.run(
        [str(worker), "-m", "pip", "install", *map(str, arguments)], check=True
    )


def checkout(destination, url, revision, *paths):
    if not (destination / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--depth=1",
                "--sparse",
                url,
                str(destination),
            ],
            check=True,
        )
    git = ["git", "-C", str(destination)]
    status = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if status.strip():
        raise RuntimeError(f"Tracked changes in upstream checkout {destination}")
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    pinned = subprocess.run(
        [*git, "rev-parse", "--verify", revision], capture_output=True, text=True
    )
    if revision != head and (pinned.returncode or pinned.stdout.strip() != head):
        subprocess.run([*git, "fetch", "--depth=1", "origin", revision], check=True)
        subprocess.run([*git, "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run([*git, "sparse-checkout", "set", *paths], check=True)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(path)


def fetch_encoder(worker, model, revision, destination=None):
    args = [model, revision, str(destination or "")]
    code = """
import sys
from huggingface_hub import snapshot_download
model, revision, directory = sys.argv[1:]
snapshot_download(model, revision=revision, local_dir=directory or None,
                  allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
"""
    env = {
        key: val
        for key, val in os.environ.items()
        if not (key.lower() == "all_proxy" and val.startswith("socks://"))
    }
    env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run([str(worker), "-c", code, *args], env=env, check=True)


def prepare_nltk():
    # nltk's current data loader requires private cache directories.
    destination = Path(
        os.getenv("APIBANK_NLTK_DATA", Path.home() / ".cache/toolcall-api-bank-nltk")
    )
    english = destination / "tokenizers/punkt_tab/english"
    names = (
        "collocations.tab",
        "sent_starters.txt",
        "abbrev_types.txt",
        "ortho_context.tab",
    )
    if all((english / name).is_file() for name in names):
        return
    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
    with urlopen(url, timeout=60) as response:
        archive = ZipFile(io.BytesIO(response.read()))
    for directory in (destination, destination / "tokenizers", english.parent, english):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in names:
        (english / name).write_bytes(archive.read(f"punkt_tab/english/{name}"))


def prepare(root, suite):
    from .cli import load_defaults

    for name in ("bfcl-v4", "api-bank", "tau2") if suite == "all" else (suite,):
        config = load_defaults(root, name, "full")
        env = root / "runs/tools" / ("bfcl" if name == "bfcl-v4" else name)
        worker = env / "bin/python"
        if not worker.is_file():
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        upstream = root / "runs/upstream"
        upstream.mkdir(parents=True, exist_ok=True)
        if name == "bfcl-v4":
            pip(
                worker,
                "bfcl-eval==" + config["version"],
                "numpy<2",
                "pandas<3",
                "soundfile",
                "sentence-transformers==6.3.0",
                "pyarrow==22.0.0",
                "datasets==4.4.1",
            )
            fetch_encoder(worker, MEMORY_ENCODER, MEMORY_ENCODER_REV)
        elif name == "api-bank":
            pins = config["official_sources"]
            checkout(
                upstream / "api-bank",
                "https://github.com/AlibabaResearch/DAMO-ConvAI.git",
                pins["repository"],
                "api-bank",
            )
            pip(
                worker,
                "rouge==1.0.1",
                "nltk==3.10.4",
                "rank-bm25==0.2.2",
                "sentence-transformers==6.3.0",
                "httpx>=0.28,<1",
            )
            for filename in (
                "level-3.json",
                "level-3-batch-inf.json",
                "level-3-batch-inf-response.json",
            ):
                download(
                    "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/"
                    f"{pins['level3_dataset']}/test-data/{filename}",
                    upstream / "api-bank-data" / filename,
                )
            encoder = upstream / "api-bank-encoder"
            weight = encoder / "model.safetensors"
            valid = False
            if weight.is_file() and (encoder / "modules.json").is_file():
                with weight.open("rb") as source:
                    valid = (
                        hashlib.file_digest(source, "sha256").hexdigest()
                        == API_BANK_ENCODER_SHA256
                    )
            if not valid:
                fetch_encoder(worker, API_BANK_ENCODER, API_BANK_ENCODER_REV, encoder)
            prepare_nltk()
        else:
            tau = upstream / "tau2-sparse"
            checkout(
                tau,
                "https://github.com/sierra-research/tau2-bench.git",
                config["git_commit"],
                "src",
                "data/tau2/domains",
                "data/tau2/user_simulator",
            )
            pip(worker, "-e", tau, "litellm==1.110.2")
        print(f"{name} ready: {worker}", flush=True)
