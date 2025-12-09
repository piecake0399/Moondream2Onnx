#!/usr/bin/env python3
# md2.py -- Moondream2 ONNX pipeline (merged decoder q4)
# Requires: onnxruntime, tokenizers, pillow, numpy

import argparse
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

# -------------------------
# Helpers
# -------------------------
def short(arr, maxlen=8):
    if arr is None:
        return "None"
    a = np.asarray(arr)
    if a.size == 0:
        return "[]"
    flat = a.flatten()
    s = ", ".join(map(str, flat[:min(len(flat), maxlen)]))
    if flat.size > maxlen:
        s += ",..."
    return f"shape={a.shape}, sample=[{s}]"

# -------------------------
# Preprocess image (CLIP-like)
# -------------------------
def preprocess_image_pil(img: Image.Image):
    img = img.convert("RGB").resize((378, 378))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # 1,C,H,W
    return arr.astype(np.float32)

# -------------------------
# Main class
# -------------------------
class Moondream2:
    def __init__(self, model_dir="weight_files", provider="CPUExecutionProvider", verbose=False):
        self.verbose = verbose

        # Load tokenizer (may fail if tokenizer.json incompatible)
        try:
            self.tokenizer = Tokenizer.from_file("tokenizer.json")
        except Exception as e:
            raise RuntimeError("Failed to load tokenizer.json with tokenizers library. "
                               "If tokenizer.json contains negative ids (Xenova ONNX), "
                               "you must provide a compatible tokenizer (vocab.json + merges or HF AutoTokenizer). "
                               f"Original error: {e}")

        # Load ONNX sessions
        self.vision = ort.InferenceSession(f"{model_dir}/vision_encoder_int8.onnx",
                                           providers=[provider])
        self.embed  = ort.InferenceSession(f"{model_dir}/embed_tokens_int8.onnx",
                                           providers=[provider])
        for i in self.embed.get_inputs(): print(i.name, i.type, i.shape)
        self.decoder= ort.InferenceSession(f"{model_dir}/decoder_model_merged_q4.onnx",
                                           providers=[provider])

        # Print decoder signature for debug
        if self.verbose:
            print("\n--- DECODER INPUT SIGNATURE ---")
            for i in self.decoder.get_inputs():
                print(i.name, i.type, i.shape)
            print("\n--- EMBED INPUT SIGNATURE ---")
            for i in self.embed.get_inputs():
                print(i.name, i.type, i.shape)
            print("")

        # Model hyperparams (match your onnx export)
        self.num_layers = 24
        self.num_heads = 32
        self.head_dim = 64
        # hidden dim from embed output (should be 2048)
        self.hidden_dim = 2048

    # -------------------------
    # Vision encoder
    # -------------------------
    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        arr = preprocess_image_pil(image)
        out = self.vision.run(None, {"pixel_values": arr})
        emb = out[0].astype(np.float32)  # expected (1, image_seq_len, hidden_dim)
        if self.verbose:
            print("[vision] output:", short(emb))
        return emb

    # -------------------------
    # Embed tokens using embed_tokens.onnx
    # -------------------------
    def embed_tokens(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids_in = ids[None, :]
        else:
            ids_in = ids
        out = self.embed.run(None, {"input_ids": ids_in})
        emb = out[0].astype(np.float32)  # (1, seq, hidden_dim)
        if self.verbose:
            print("[embed] in", ids_in.shape, "->", short(emb))
        return emb

    # -------------------------
    # init zero-length past kv
    # -------------------------
    def init_past(self):
        empty = np.zeros((1, self.num_heads, 0, self.head_dim), dtype=np.float32)
        past = {}
        for i in range(self.num_layers):
            past[f"past_key_values.{i}.key"] = empty
            past[f"past_key_values.{i}.value"] = empty
        return past

    # -------------------------
    # greedy decode for merged decoder (first call: full inputs_embeds; subsequent calls: 1 token + past)
    # -------------------------
    def greedy_decode(self, prompt_ids, image_embeds, max_new_tokens=60):
        """
        prompt_ids: 1D list/np.array of prompt token ids (no batch)
        image_embeds: (1, image_seq_len, hidden_dim) from vision encoder
        """
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        prompt_embeds = self.embed_tokens(prompt_ids)  # (1, P, H)

        # First call: concatenate image + prompt into full inputs_embeds
        full_embeds = np.concatenate([image_embeds, prompt_embeds], axis=1)  # (1, S, H)
        seq_len = int(full_embeds.shape[1])

        # Prepare initial inputs for first forward
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # empty past
        past = self.init_past()

        # Build inputs dict for first call
        inputs = {
            "inputs_embeds": full_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        inputs.update(past)

        if self.verbose:
            print("[decode] First forward shapes:",
                "inputs_embeds", short(full_embeds),
                "attention_mask", attention_mask.shape,
                "position_ids", position_ids.shape)

        # First forward: get logits & present
        outputs = self.decoder.run(None, inputs)
        logits = outputs[0]  # (1, seq_len, vocab) or (1, seq_len, vocab)
        # outputs[1:] are interleaved present.N.key, present.N.value
        present = outputs[1:]

        # parse present into past dict (map to past_key_values.i.key/value)
        new_past = {}
        idx = 0
        for i in range(self.num_layers):
            new_past[f"past_key_values.{i}.key"] = present[idx]
            new_past[f"past_key_values.{i}.value"] = present[idx + 1]
            idx += 2
        past = new_past

        # choose next token from last logit (logit for last position of full_embeds)
        if logits.ndim == 3:
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        elif logits.ndim == 2:
            next_id = int(np.argmax(logits, axis=-1)[0])
        else:
            raise RuntimeError("unexpected logits shape: " + str(logits.shape))

        generated = [next_id]

        # ===== FIXED: After first forward, cached length = seq_len (NOT 1) =====
        past_len = seq_len           # <-- sửa chỗ này
        # Next token absolute position = seq_len (0-based)
        next_pos = seq_len

        if self.verbose:
            print(f"[decode] step0 next_id={next_id}, seq_len={seq_len}, past_len={past_len}")

        # Subsequent steps: feed one token embedding and use past
        for step in range(max_new_tokens - 1):
            # embed last token
            token_id_arr = np.array([next_id], dtype=np.int64)
            token_emb = self.embed_tokens(token_id_arr)  # (1,1,H)

            # attention mask length = past_len + 1  (past_len reflects full cached length)
            attention_mask = np.ones((1, past_len + 1), dtype=np.int64)
            # position_ids for token we are feeding (shape 1x1)
            position_ids = np.array([[next_pos]], dtype=np.int64)

            ort_inputs = {
                "inputs_embeds": token_emb.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }
            # attach past_kv
            ort_inputs.update(past)

            if self.verbose:
                print(f"[decode] step={step+1} feed token {next_id} pos={next_pos} attn={attention_mask.shape}")

            outs = self.decoder.run(None, ort_inputs)
            logits = outs[0]
            # parse present
            present = outs[1:]
            # map present -> past
            new_past = {}
            idx = 0
            for i in range(self.num_layers):
                new_past[f"past_key_values.{i}.key"] = present[idx]
                new_past[f"past_key_values.{i}.value"] = present[idx + 1]
                idx += 2
            past = new_past

            # next id from logits last token
            if logits.ndim == 3:
                next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            elif logits.ndim == 2:
                next_id = int(np.argmax(logits, axis=-1)[0])
            else:
                raise RuntimeError("unexpected logits shape: " + str(logits.shape))

            generated.append(next_id)

            # update counters
            past_len += 1   # cached length increases by 1
            next_pos += 1

            # eos check
            try:
                eos = self.tokenizer.token_to_id("</s>")
            except Exception:
                eos = None
            if eos is not None and next_id == eos:
                if self.verbose:
                    print("[decode] EOS generated, stop.")
                break

        return generated


    # -------------------------
    # run tasks (Florence2-style buckets)
    # -------------------------
    def run_tasks(self, image_input, tasks, max_new_tokens=60):
        """
        tasks: list of dicts like {"name":"caption", "prompt":"Describe this image."}
        returns dict name -> generated_text
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input

        image_embeds = self.encode_image(img)

        results = {}
        for t in tasks:
            prompt = t["prompt"]
            prompt_ids = np.array(self.tokenizer.encode(prompt).ids, dtype=np.int64)
            gen_ids = self.greedy_decode(prompt_ids, image_embeds, max_new_tokens=max_new_tokens)
            all_ids = prompt_ids.tolist() + gen_ids
            txt = self.tokenizer.decode(all_ids)
            results[t["name"]] = txt
        return results

    # -------------------------
    # convenience describe
    # -------------------------
    def describe(self, image_input, question="Describe this image.", max_new_tokens=60):
        tasks = [{"name":"caption", "prompt": f"<Image>\n{question}\nAnswer:"}]
        res = self.run_tasks(image_input, tasks, max_new_tokens=max_new_tokens)
        return res["caption"]

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--model-dir", default="weight_files")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--task", choices=["caption","vqa","all"], default="caption")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args()

    # prepare tasks
    if args.task == "caption":
        tasks = [{"name":"caption", "prompt":"<Image>\nDescribe this image.\nAnswer:"}]
    elif args.task == "vqa":
        tasks = [
            {"name":"what_is_in_image", "prompt":"<Image>\nWhat is in the image?\nAnswer:"},
            {"name":"color_question", "prompt":"<Image>\nWhat is the dominant color?\nAnswer:"}
        ]
    else:
        tasks = [
            {"name":"caption", "prompt":"<Image>\nDescribe this image.\nAnswer:"},
            {"name":"what_is_in_image", "prompt":"<Image>\nWhat is in the image?\nAnswer:"},
            {"name":"vqa_example", "prompt":"<Image>\nIs there a car in the image?\nAnswer:"},
        ]

    model = Moondream2(model_dir=args.model_dir, provider=args.provider, verbose=args.verbose)

    start = time.time()
    results = model.run_tasks(args.image, tasks, max_new_tokens=args.max_tokens)
    end = time.time()

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"-- {k} --")
        print(v)
        print()

    print(f"Elapsed {end - start:.2f}s")

if __name__ == "__main__":
    main()
