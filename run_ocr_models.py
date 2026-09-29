#!/usr/bin/env python3
"""
Esegue una batteria di modelli Vision/OCR tramite l'API locale di Ollama.

Struttura prodotta:

output/
└── experiment_001/
    ├── experiment.json
    ├── glm-ocr_latest/
    │   ├── prompt.txt
    │   ├── metadata.jsonl
    │   ├── libretto_001.txt
    │   └── ...
    ├── deepseek-ocr_latest/
    │   └── ...
    └── ...

Scelte metodologiche:
- ogni esecuzione normale crea un nuovo esperimento numerato;
- tutti i modelli eseguiti nello stesso avvio appartengono allo stesso esperimento;
- i file TXT contengono esattamente il testo restituito dal modello;
- non viene applicata alcuna pulizia o normalizzazione;
- non viene creata una cartella raw, perché il file principale è già l'output grezzo;
- il prompt effettivo viene salvato dentro la cartella di ogni modello;
- i parametri comuni vengono salvati una sola volta in experiment.json;
- per riprendere un esperimento esistente si usa --experiment-id N.

Non richiede pacchetti Python esterni.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "screenshot"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_MODELS = [
    "glm-ocr:latest",
    "deepseek-ocr:latest",
    "qwen3-vl:32b-instruct",
    "gemma4:31b",
    "nemotron3:33b",
    "qwen3.5:27b",
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

# Modifica direttamente questo prompt quando vuoi creare un nuovo esperimento.
# Lo script ne salva automaticamente una copia nella cartella di ogni
# modello generalista eseguito.
OCR_PROMPT = """Extract all the text visible in the image.

The image contains a historical document written in Italian.

Return only the transcription, without introductions, explanations, comments, descriptions, or Markdown."""

# Prompt minimi richiesti dai modelli OCR specializzati.
MODEL_PROMPTS = {
    "glm-ocr:latest": "Text Recognition:",
    "deepseek-ocr:latest": "Extract the text in the image.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trascrive tutte le immagini con più modelli Ollama e salva "
            "i risultati in una cartella numerata per esperimento."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Modelli da eseguire. Predefiniti: %(default)s",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Cartella contenente le immagini.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella radice degli esperimenti.",
    )
    parser.add_argument(
        "--experiment-id",
        "--run-id",
        dest="experiment_id",
        type=int,
        default=None,
        help=(
            "Numero di un esperimento esistente da riprendere. "
            "Se omesso, viene creato automaticamente il successivo."
        ),
    )
    parser.add_argument(
        "--description",
        default="",
        help="Breve descrizione salvata in experiment.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Elabora soltanto le prime N immagini.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rigenera i TXT già esistenti nello stesso esperimento. "
            "I nuovi metadati vengono aggiunti a metadata.jsonl."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout massimo per una singola richiesta, in secondi.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Dimensione del contesto Ollama.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=4096,
        help="Numero massimo di token generati.",
    )
    parser.add_argument(
        "--keep-alive",
        default="30m",
        help="Tempo per cui il modello resta caricato durante la batteria.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_key(path: Path) -> list[Any]:
    """Ordina libretto_2 prima di libretto_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def model_folder_name(model: str) -> str:
    """Converte il tag Ollama in un nome di cartella portabile."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.replace(":", "_"))


def experiment_folder_name(experiment_id: int) -> str:
    return f"experiment_{experiment_id:03d}"


def find_next_experiment_id(output_root: Path) -> int:
    """
    Cerca cartelle del tipo experiment_001 e restituisce il numero successivo.
    Le vecchie cartelle organizzate per modello vengono ignorate.
    """
    pattern = re.compile(r"^experiment_(\d+)$")
    ids: list[int] = []

    if output_root.exists():
        for child in output_root.iterdir():
            if not child.is_dir():
                continue

            match = pattern.fullmatch(child.name)
            if match:
                ids.append(int(match.group(1)))

    return max(ids, default=0) + 1


def api_json(
    endpoint: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Errore HTTP {exc.code} restituito da Ollama: {body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Ollama non è raggiungibile su {OLLAMA_BASE_URL}. "
            "Avvialo in un altro terminale con: ollama serve"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama ha restituito una risposta JSON non valida: {raw[:500]}"
        ) from exc


def installed_models() -> set[str]:
    response = api_json("/api/tags")
    return {
        model["name"]
        for model in response.get("models", [])
        if isinstance(model.get("name"), str)
    }


def list_images(image_dir: Path, limit: int | None) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Cartella immagini non trovata: {image_dir}")

    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit deve essere maggiore di zero.")
        images = images[:limit]

    if not images:
        raise FileNotFoundError(f"Nessuna immagine trovata in: {image_dir}")

    return images


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def prompt_for_model(model: str) -> str:
    return MODEL_PROMPTS.get(model, OCR_PROMPT)


def ns_to_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) / 1_000_000_000, 6)
    return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def create_or_validate_experiment(
    *,
    experiment_folder: Path,
    experiment_id: int,
    description: str,
    models: list[str],
    images: list[Path],
    image_dir: Path,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    timeout: int,
) -> dict[str, Any]:
    """
    Crea experiment.json oppure verifica che un esperimento ripreso
    utilizzi gli stessi parametri e le stesse immagini.
    """
    manifest_path = experiment_folder / "experiment.json"
    image_names = [path.name for path in images]

    if not manifest_path.exists():
        manifest = {
            "experiment_id": experiment_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "description": description,
            "image_directory": relative_or_absolute(image_dir),
            "images": image_names,
            "number_of_images": len(image_names),
            "models": list(dict.fromkeys(models)),
            "parameters": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "keep_alive": keep_alive,
                "timeout_seconds": timeout,
                "temperature": 0,
                "seed": 42,
                "stream": False,
                "think": False,
            },
            "summary": {
                "completed": 0,
                "skipped": 0,
                "errors": 0,
            },
        }
        write_json(manifest_path, manifest)
        return manifest

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Impossibile leggere correttamente {manifest_path}"
        ) from exc

    expected_parameters = {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "keep_alive": keep_alive,
        "timeout_seconds": timeout,
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "think": False,
    }

    differences: list[str] = []

    if manifest.get("image_directory") != relative_or_absolute(image_dir):
        differences.append("cartella immagini")

    if manifest.get("images") != image_names:
        differences.append("insieme o ordine delle immagini")

    if manifest.get("parameters") != expected_parameters:
        differences.append("parametri Ollama")

    saved_description = manifest.get("description", "")
    if description and saved_description and description != saved_description:
        differences.append("descrizione")

    if differences:
        raise RuntimeError(
            "L'esperimento esistente usa impostazioni diverse: "
            + ", ".join(differences)
            + ". Crea un nuovo esperimento senza --experiment-id."
        )

    if description and not saved_description:
        manifest["description"] = description

    saved_models = manifest.get("models", [])
    manifest["models"] = list(dict.fromkeys([*saved_models, *models]))
    manifest["updated_at"] = utc_now()
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    return manifest


def update_experiment_summary(
    experiment_folder: Path,
    totals: dict[str, int],
) -> None:
    manifest_path = experiment_folder / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["updated_at"] = utc_now()
    manifest["status"] = (
        "completed" if totals["errors"] == 0 else "completed_with_errors"
    )
    manifest["summary"] = totals

    write_json(manifest_path, manifest)


def transcribe_image(
    *,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_to_base64(image_path)],
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    started = time.perf_counter()
    response = api_json(
        "/api/chat",
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    wall_seconds = time.perf_counter() - started

    message = response.get("message") or {}
    content = message.get("content")

    if not isinstance(content, str):
        raise RuntimeError(
            f"Ollama non ha restituito message.content: {response}"
        )

    metrics = {
        "wall_seconds": round(wall_seconds, 3),
        "total_duration_seconds": ns_to_seconds(
            response.get("total_duration")
        ),
        "load_duration_seconds": ns_to_seconds(
            response.get("load_duration")
        ),
        "prompt_eval_duration_seconds": ns_to_seconds(
            response.get("prompt_eval_duration")
        ),
        "eval_duration_seconds": ns_to_seconds(
            response.get("eval_duration")
        ),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "done_reason": response.get("done_reason"),
    }

    # Nessuna pulizia, strip, normalizzazione o aggiunta di newline:
    # viene restituito esattamente message.content ricevuto da Ollama.
    return content, metrics


def unload_model(model: str) -> None:
    """Libera la memoria occupata dal modello appena concluso."""
    try:
        api_json(
            "/api/generate",
            method="POST",
            payload={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=120,
        )
    except Exception as exc:
        print(
            f"  Attenzione: impossibile scaricare {model}: {exc}",
            file=sys.stderr,
        )


def prepare_model_folder(
    *,
    experiment_folder: Path,
    model: str,
    prompt: str,
) -> Path:
    folder = experiment_folder / model_folder_name(model)
    folder.mkdir(parents=True, exist_ok=True)

    prompt_path = folder / "prompt.txt"

    if prompt_path.exists():
        saved_prompt = prompt_path.read_text(encoding="utf-8")
        if saved_prompt != prompt:
            raise RuntimeError(
                f"La cartella {folder} contiene già un prompt diverso. "
                "Non è possibile mescolare due prompt nello stesso esperimento."
            )
    else:
        # Anche il prompt viene salvato senza aggiungere caratteri.
        prompt_path.write_text(prompt, encoding="utf-8")

    return folder


def run_model(
    *,
    model: str,
    experiment_id: int,
    experiment_folder: Path,
    images: list[Path],
    overwrite: bool,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
) -> dict[str, int]:
    prompt = prompt_for_model(model)
    folder = prepare_model_folder(
        experiment_folder=experiment_folder,
        model=model,
        prompt=prompt,
    )
    log_path = folder / "metadata.jsonl"

    counts = {"completed": 0, "skipped": 0, "errors": 0}

    print(f"\n=== Modello: {model} ===")
    print(f"Esperimento: {experiment_folder.name}")
    print(f"Output: {folder}")
    print(f"Immagini: {len(images)}")

    for index, image_path in enumerate(images, start=1):
        output_path = folder / f"{image_path.stem}.txt"

        if output_path.exists() and not overwrite:
            counts["skipped"] += 1
            print(f"[{index}/{len(images)}] SKIP  {image_path.name}")
            continue

        print(f"[{index}/{len(images)}] OCR   {image_path.name}", flush=True)
        created_at = utc_now()

        try:
            transcription, metrics = transcribe_image(
                model=model,
                image_path=image_path,
                prompt=prompt,
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=num_predict,
                keep_alive=keep_alive,
            )

            # newline="" evita la conversione automatica dei ritorni a capo
            # sui sistemi in cui Python potrebbe applicarla.
            with output_path.open("w", encoding="utf-8", newline="") as file:
                file.write(transcription)

            record = {
                "created_at": created_at,
                "status": "completed",
                "experiment_id": experiment_id,
                "model": model,
                "image": image_path.name,
                "output_file": relative_or_absolute(output_path),
                "characters": len(transcription),
                "lines": len(transcription.splitlines()),
                **metrics,
            }
            append_jsonl(log_path, record)

            counts["completed"] += 1
            print(
                f"             OK in {metrics['wall_seconds']:.1f}s "
                f"({len(transcription)} caratteri)"
            )

        except Exception as exc:
            counts["errors"] += 1
            append_jsonl(
                log_path,
                {
                    "created_at": created_at,
                    "status": "error",
                    "experiment_id": experiment_id,
                    "model": model,
                    "image": image_path.name,
                    "error": str(exc),
                },
            )
            print(f"             ERRORE: {exc}", file=sys.stderr)

    unload_model(model)
    return counts


def main() -> int:
    args = parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    try:
        available = installed_models()
        images = list_images(image_dir, args.limit)
    except Exception as exc:
        print(f"Errore iniziale: {exc}", file=sys.stderr)
        return 1

    missing = [model for model in args.models if model not in available]
    if missing:
        print("Modelli richiesti non installati:", file=sys.stderr)
        for model in missing:
            print(f"  - {model}", file=sys.stderr)
        print("\nControlla i nomi con: ollama list", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    if args.experiment_id is not None:
        if args.experiment_id < 1:
            print(
                "Errore: --experiment-id deve essere maggiore di zero.",
                file=sys.stderr,
            )
            return 1

        experiment_id = args.experiment_id
        mode = "ripresa manuale"
    else:
        experiment_id = find_next_experiment_id(output_root)
        mode = "creato automaticamente"

    experiment_folder = (
        output_root / experiment_folder_name(experiment_id)
    )
    experiment_folder.mkdir(parents=True, exist_ok=True)

    try:
        create_or_validate_experiment(
            experiment_folder=experiment_folder,
            experiment_id=experiment_id,
            description=args.description,
            models=args.models,
            images=images,
            image_dir=image_dir,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            keep_alive=args.keep_alive,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Errore nella configurazione: {exc}", file=sys.stderr)
        return 1

    print("Batteria OCR Ollama")
    print(f"Esperimento: {experiment_folder.name} ({mode})")
    print(f"Immagini:    {image_dir}")
    print(f"Output:      {experiment_folder}")
    print(f"Modelli:     {len(args.models)}")
    print(f"File:        {len(images)}")
    print(f"Contesto:    {args.num_ctx}")
    print(f"Num predict: {args.num_predict}")
    print(
        "Resume:      "
        + ("no, sovrascrive" if args.overwrite else "sì, salta esistenti")
    )

    totals = {"completed": 0, "skipped": 0, "errors": 0}

    for model in args.models:
        try:
            result = run_model(
                model=model,
                experiment_id=experiment_id,
                experiment_folder=experiment_folder,
                images=images,
                overwrite=args.overwrite,
                timeout=args.timeout,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive,
            )
        except Exception as exc:
            print(
                f"Errore nella preparazione di {model}: {exc}",
                file=sys.stderr,
            )
            totals["errors"] += len(images)
            continue

        for key in totals:
            totals[key] += result[key]

    try:
        update_experiment_summary(experiment_folder, totals)
    except Exception as exc:
        print(
            f"Attenzione: impossibile aggiornare experiment.json: {exc}",
            file=sys.stderr,
        )

    print("\n=== Riepilogo finale ===")
    print(f"Esperimento: {experiment_folder.name}")
    print(f"Completati:  {totals['completed']}")
    print(f"Saltati:     {totals['skipped']}")
    print(f"Errori:      {totals['errors']}")

    return 0 if totals["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
