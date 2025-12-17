#!/usr/bin/env python3
# md2.py -- Moondream2 ONNX pipeline (merged decoder q4)
# Requires: onnxruntime, tokenizers, pillow, numpy, tqdm, datasets

import argparse
import time
import re
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

import psutil

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
# Parse bbox text robustly
# -------------------------
_bbox_re = re.compile(
    r"\[\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\]"
)

def parse_bboxes_from_text(text):
    """
    Trả về list các bbox ở dạng normalized floats: [{"x_min":..., "y_min":..., "x_max":..., "y_max":...}, ...]
    Lấy mọi occurences của pattern [x,y,x,y]
    """
    if not isinstance(text, str):
        return []
    matches = _bbox_re.findall(text)
    bboxes = []
    for m in matches:
        try:
            nums = [float(x) for x in m]
            # sanity clamp 0..1
            nums = [max(0.0, min(1.0, v)) for v in nums]
            bboxes.append({
                "x_min": nums[0],
                "y_min": nums[1],
                "x_max": nums[2],
                "y_max": nums[3],
            })
        except Exception:
            continue
    return bboxes

# -------------------------
# IoU computation
# -------------------------
def compute_iou(boxA, boxB):
    """boxA, boxB dạng [x1,y1,x2,y2] tuyệt đối"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

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
                               f"Original error: {e}")

        # Load ONNX sessions (FP16 vision/embed + merged decoder Q4)
        self.vision = ort.InferenceSession(f"{model_dir}/vision_encoder_fp16.onnx",
                                           providers=[provider])
        self.embed  = ort.InferenceSession(f"{model_dir}/embed_tokens_fp16.onnx",
                                           providers=[provider])
        self.decoder= ort.InferenceSession(f"{model_dir}/decoder_model_merged_q4.onnx",
                                           providers=[provider])

        # Print signatures if verbose
        if self.verbose:
            print("\n--- DECODER INPUT SIGNATURE ---")
            for i in self.decoder.get_inputs():
                print(i.name, i.type, i.shape)
            print("\n--- EMBED INPUT SIGNATURE ---")
            for i in self.embed.get_inputs():
                print(i.name, i.type, i.shape)
            print("")

        # Model hyperparams (tune if differs)
        self.num_layers = 24
        self.num_heads = 32
        self.head_dim = 64
        self.hidden_dim = 2048  # embed output

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
    # greedy decode (unchanged, returns list of token ids generated)
    # -------------------------
    def greedy_decode(self, prompt_ids, image_embeds, max_new_tokens=60):
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        prompt_embeds = self.embed_tokens(prompt_ids)  # (1, P, H)

        # First call: concat image + prompt
        full_embeds = np.concatenate([image_embeds, prompt_embeds], axis=1)  # (1, S, H)
        seq_len = int(full_embeds.shape[1])

        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        past = self.init_past()

        inputs = {
            "inputs_embeds": full_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        inputs.update(past)

        outputs = self.decoder.run(None, inputs)
        logits = outputs[0]
        present = outputs[1:]

        # parse present -> past
        new_past = {}
        idx = 0
        for i in range(self.num_layers):
            new_past[f"past_key_values.{i}.key"] = present[idx]
            new_past[f"past_key_values.{i}.value"] = present[idx + 1]
            idx += 2
        past = new_past

        # pick token
        if logits.ndim == 3:
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        elif logits.ndim == 2:
            next_id = int(np.argmax(logits, axis=-1)[0])
        else:
            raise RuntimeError("unexpected logits shape: " + str(logits.shape))

        generated = [next_id]
        past_len = seq_len
        next_pos = seq_len

        # subsequent
        for step in range(max_new_tokens - 1):
            token_id_arr = np.array([next_id], dtype=np.int64)
            token_emb = self.embed_tokens(token_id_arr)  # (1,1,H)

            attention_mask = np.ones((1, past_len + 1), dtype=np.int64)
            position_ids = np.array([[next_pos]], dtype=np.int64)

            ort_inputs = {
                "inputs_embeds": token_emb.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }
            ort_inputs.update(past)

            outs = self.decoder.run(None, ort_inputs)
            logits = outs[0]
            present = outs[1:]

            # map present -> past
            new_past = {}
            idx = 0
            for i in range(self.num_layers):
                new_past[f"past_key_values.{i}.key"] = present[idx]
                new_past[f"past_key_values.{i}.value"] = present[idx + 1]
                idx += 2
            past = new_past

            if logits.ndim == 3:
                next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            elif logits.ndim == 2:
                next_id = int(np.argmax(logits, axis=-1)[0])
            else:
                raise RuntimeError("unexpected logits shape: " + str(logits.shape))

            generated.append(next_id)
            past_len += 1
            next_pos += 1

            # quick EOS check
            try:
                eos = self.tokenizer.token_to_id("</s>")
            except Exception:
                eos = None
            if eos is not None and next_id == eos:
                break

        return generated

    # -------------------------
    # run tasks (with special handling for visual grounding)
    # -------------------------
    def run_tasks(self, image_input, tasks, max_new_tokens=60):
        """
        tasks: list of dicts like {"name":"caption", "prompt":"Describe this image."}
        returns dict name -> output (string or structured)
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input

        image_embeds = self.encode_image(img)
        results = {}

        for t in tasks:
            name = t["name"]
            prompt = t["prompt"]

            # Tokenize prompt
            prompt_ids = np.array(self.tokenizer.encode(prompt).ids, dtype=np.int64)

            # Generate token ids
            gen_ids = self.greedy_decode(prompt_ids, image_embeds, max_new_tokens=max_new_tokens)

            # Decode generated tokens to text (only generated part)
            gen_text = self.tokenizer.decode(gen_ids)

            # Full text (prompt + generated) if needed
            full_text = self.tokenizer.decode(prompt_ids.tolist() + gen_ids)

            # If this is a visual grounding task, parse bbox(es) and return structured result
            if "ground" in name or "visual" in name or "phrase" in name:
                # parse bboxes from gen_text (prefer generated only), fallback to full_text
                bboxes = parse_bboxes_from_text(gen_text)
                if not bboxes:
                    bboxes = parse_bboxes_from_text(full_text)

                # take ONLY the first bbox as canonical answer (to avoid repeated duplicates)
                objects = []
                if bboxes:
                    first = bboxes[0]
                    objects.append(first)

                # Build structured result
                results[name] = {
                    "text": full_text,
                    "objects": objects  # normalized coords
                }
            else:
                # generic text task
                results[name] = full_text

            if self.verbose:
                print(f"[run_tasks] task={name} -> text(len)={len(full_text)} bboxes_found={len(results.get(name,{} ).get('objects',[])) if isinstance(results[name], dict) else 0}")

        return results

    # -------------------------
    # convenience describe
    # -------------------------
    def describe(self, image_input, question="Describe this image.", max_new_tokens=60):
        tasks = [{"name":"caption", "prompt": f"<Image>\n{question}\nAnswer:"}]
        res = self.run_tasks(image_input, tasks, max_new_tokens=max_new_tokens)
        return res["caption"]

# -------------------------
# Evaluation helper (RefCOCO)
# -------------------------
def evaluate_dataset_moondream2(model, dataset, n_samples=None):
    total = 0
    correct = 0
    processed_samples = 0

    infers = []
    
    process = psutil.Process()
    rss_before = process.memory_info().rss

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating RefCOCO with Moondream2")):
        if (n_samples is not None) and (processed_samples >= n_samples):
            break

        img = sample["image"].convert("RGB")
        ref_list = sample["ref_list"]

        for ref_info in ref_list:
            ann_info = ref_info["ann_info"]
            gt_bbox = ann_info["bbox"]  # COCO format: [x, y, w, h]
            x, y, w, h = gt_bbox
            gt = [x, y, x + w, y + h]  # [x1, y1, x2, y2]

            sentences = ref_info["ref_info"]["sentences"]
            for sentence_info in sentences:
                expr = sentence_info["sent"]

                # compose strong JSON-only prompt (ask for exactly one bbox)
                prompt = (
                    "<Image>\n"
                    "Locate the following object and output EXACTLY ONE bounding box in JSON array format:\n"
                    f"\"{expr}\"\n"
                    "Output format (required): [{'[x_min, y_min, x_max, y_max]'}]\n"
                    "Only output the JSON array and nothing else.\n"
                    "Answer:"
                )

                try:
                    # -----------MEASURE------------
                    start = time.time()
                    # ------------------------------
                    result = model.run_tasks(
                        img,
                        tasks=[{"name":"visual_grounding", "prompt": prompt}],
                        max_new_tokens=48
                    )
                    objects = result["visual_grounding"]["objects"]
                    # -----------MEASURE------------
                    end = time.time()
                    # ------------------------------
                    infer_time = end - start
                    infers.append(infer_time)
                except Exception:
                    objects = []

                if not objects:
                    total += 1
                    processed_samples += 1
                    continue

                img_w, img_h = img.size
                best_iou = 0.0
                for obj in objects:
                    x1 = obj["x_min"] * img_w
                    y1 = obj["y_min"] * img_h
                    x2 = obj["x_max"] * img_w
                    y2 = obj["y_max"] * img_h
                    pred_box = [x1, y1, x2, y2]

                    iou = compute_iou(pred_box, gt)
                    best_iou = max(best_iou, iou)

                if best_iou >= 0.5:
                    correct += 1
                total += 1
                processed_samples += 1

    acc = correct / total if total > 0 else 0.0
    rss_after = process.memory_info().rss
    total_mem_used_bytes = rss_after - rss_before
    total_mem_used_mb = total_mem_used_bytes / 1024 / 1024

    print("------- Evaluation Results ------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {acc*100:.2f}%")
    print("---------------------------------")
    print(f"Average inference time: {np.mean(infers):.4f} seconds")
    print(f"Minimum inference time: {np.min(infers):.4f} seconds")
    print(f"Maximum inference time: {np.max(infers):.4f} seconds")
    print("---------------------------------")
    print(f"Total memory used during benchmark: {total_mem_used_mb:.2f} MB")
    # print(f"Maximum peak RAM usage: {np.max(peak_mems) / 1024 / 1024:.2f} MB")
    print("---------------------------------")
    #return {"accuracy": acc, "correct": correct, "total": total}

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--model-dir", default="weight_files")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--task", choices=["caption","vqa","visual_grounding","all"], default="caption")
    parser.add_argument("--expr", default=None, help="expression for grounding")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args()

    if args.task == "caption":
        tasks = [{"name":"caption", "prompt":"<Image>\nDescribe this image.\nAnswer:"}]
    elif args.task == "vqa":
        tasks = [
            {"name":"what_is_in_image", "prompt":"<Image>\nWhat is in the image?\nAnswer:"},
            {"name":"color_question", "prompt":"<Image>\nWhat is the dominant color?\nAnswer:"}
        ]
    elif args.task == "visual_grounding":
        expr = args.expr or "the object"
        prompt = (
            "<Image>\n"
            "Locate the following object and output EXACTLY ONE bounding box in JSON array format:\n"
            f"\"{expr}\"\n"
            "Output format (required): [[x_min, y_min, x_max, y_max]]\n"
            "Only output the JSON array and nothing else.\n"
            "Answer:"
        )
        tasks = [{"name":"visual_grounding", "prompt": prompt}]
    else:
        tasks = [
            {"name":"caption", "prompt":"<Image>\nDescribe this image.\nAnswer:"},
            {"name":"what_is_in_image", "prompt":"<Image>\nWhat is in the image?\nAnswer:"},
            {"name":"visual_grounding", "prompt":"<Image>\nLocate the following object and output EXACTLY ONE bounding box in JSON array format:\n\"the object\"\nOutput format (required): [[x_min, y_min, x_max, y_max]]\nOnly output the JSON array and nothing else.\nAnswer:"}
        ]

    model = Moondream2(model_dir=args.model_dir, provider=args.provider, verbose=args.verbose)

    start = time.time()
    results = model.run_tasks(args.image, tasks, max_new_tokens=args.max_tokens)
    end = time.time()

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"-- {k} --")
        print(json.dumps(v, indent=2) if isinstance(v, dict) else v)
        print()

    print(f"Elapsed {end - start:.2f}s")

if __name__ == "__main__":
    # main()
    model = Moondream2(model_dir="weight_files", provider="CPUExecutionProvider", verbose=False)
    dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")
    evaluate_dataset_moondream2(model, dataset, n_samples=20)
    
    # img = Image.open("spaceshuttle.jpg")
    # img_w, img_h = img.size
    # gt = [
    #     0.39 * img_w,
    #     0.11 * img_h,
    #     0.56 * img_w,
    #     0.75 * img_h,
    # ]

    # expr = "the space shuttle"
    # prompt = (
    #     "<Image>\n"
    #     "Locate the following object and output EXACTLY ONE bounding box in JSON array format:\n"
    #     f"\"{expr}\"\n"
    #     "Output format (required): [[x_min, y_min, x_max, y_max]]\n"
    #     "Only output the JSON array and nothing else.\n"
    #     "Answer:"
    # )
    # tasks = [{"name":"visual_grounding", "prompt": prompt}]
    # result = model.run_tasks(
    #     "spaceshuttle.jpg",
    #     tasks=tasks,
    #     max_new_tokens=30
    # )
    # objects = result["visual_grounding"]["objects"]
    # obj = objects[0]

    # # normalized → pixel
    # pred = [
    #     obj["x_min"] * img_w,
    #     obj["y_min"] * img_h,
    #     obj["x_max"] * img_w,
    #     obj["y_max"] * img_h,
    # ]

    # iou = compute_iou(pred, gt)
    # print("Pred bbox:", pred)
    # print("IoU =", iou)

