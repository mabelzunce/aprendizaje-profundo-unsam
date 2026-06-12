"""
gpt2_generate.py
----------------
Carga GPT-2 desde HuggingFace (openai-community/gpt2) y genera texto
de forma interactiva desde la consola.

Uso
---
    python gpt2_generate.py
    python gpt2_generate.py --model gpt2-medium
    python gpt2_generate.py --max_new_tokens 200 --temperature 0.8 --top_k 50

Dependencias
------------
    pip install transformers torch
"""

import argparse
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ── Argumentos ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 interactive text generation")
    p.add_argument(
        "--model",
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Variante de GPT-2 a cargar (default: gpt2 ~500 MB).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Máximo de tokens nuevos a generar (default: 150).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature de muestreo: <1 más conservador, >1 más aleatorio (default: 0.9).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling: limita el muestreo a los k tokens más probables (default: 50).",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95).",
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penaliza repetir tokens ya generados. 1.0 = sin penalización (default: 1.1).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Dispositivo: 'cpu', 'cuda', 'mps'. Si no se indica se detecta automáticamente.",
    )
    return p.parse_args()


# ── Utilidades ───────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    print(f"\n📦 Cargando tokenizador y modelo '{model_name}'…")
    print("   (la primera vez se descargan ~500 MB; las siguientes se usan desde caché)\n")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo cargado en '{device}'.")
    print(f"   Parámetros: {n_params:,}")
    print(f"   Vocabulario: {tokenizer.vocab_size:,} tokens")
    return tokenizer, model


def generate(
    prompt: str,
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,  # evita warning de padding
        )

    # Decodificamos sólo los tokens nuevos (no el prompt)
    new_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated_text


# ── Loop interactivo ─────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          GPT-2 — Generación interactiva de texto         ║
║  Escribí un prompt y presioná Enter para generar texto.  ║
║  Comandos especiales:                                    ║
║    :q  o  exit  → salir                                  ║
║    :config      → mostrar parámetros actuales            ║
╚══════════════════════════════════════════════════════════╝
"""


def print_config(args: argparse.Namespace) -> None:
    print("\n── Configuración actual ──────────────────────────────────")
    print(f"  model             : {args.model}")
    print(f"  max_new_tokens    : {args.max_new_tokens}")
    print(f"  temperature       : {args.temperature}")
    print(f"  top_k             : {args.top_k}")
    print(f"  top_p             : {args.top_p}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print("──────────────────────────────────────────────────────────\n")


def interactive_loop(
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    device: str,
    args: argparse.Namespace,
) -> None:
    print(BANNER)
    print_config(args)

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego.")
            sys.exit(0)

        if not prompt:
            continue

        if prompt.lower() in (":q", "exit", "quit"):
            print("👋 Hasta luego.")
            sys.exit(0)

        if prompt == ":config":
            print_config(args)
            continue

        print("\n── Texto generado ────────────────────────────────────────")
        print(f"{prompt}", end="")  # mostramos el prompt en la misma línea

        try:
            continuation = generate(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            print(continuation)
        except Exception as e:
            print(f"\n❌ Error al generar: {e}")

        print("──────────────────────────────────────────────────────────\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = args.device or detect_device()
    tokenizer, model = load_model(args.model, device)
    interactive_loop(tokenizer, model, device, args)


if __name__ == "__main__":
    main()
