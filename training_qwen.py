# ✅ Clean install of all required packages
#!pip install  datasets transformers peft accelerate bitsandbytes torchmetrics
# code with qwen

# dependencies for tikz code -> pdf -> images 
# sudo apt-get update
# sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-pictures poppler-utils
# pip install pdf2image


# 🧠 Step 2: Import modules
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from copy import deepcopy

from transformers import (
    CLIPProcessor,
    CLIPModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.multimodal import CLIPScore as TorchCLIPScore
from qwen_vl_utils import process_vision_info


save_dir = "/home/mila/d/daria.yasafova/scratch/tikz_project/iangola"
HUGGINGFACE_CACHE_DIR = "/home/mila/d/daria.yasafova/scratch/huggingface"


from collections import Counter
from nltk.util import ngrams
import numpy as np

# for tikz code -> pdf -> image
import subprocess
import tempfile
from pdf2image import convert_from_path
from PIL import Image


def crystal_bleu(candidate_list, references_list, n=4):
    """
    Simplified CrystalBLEU for evaluating text generation (e.g., TikZ).
    Based on n-gram overlap, with smoothing.
    """

    def count_ngrams(sequence, n):
        return Counter(ngrams(sequence, n)) if len(sequence) >= n else Counter()

    scores = []
    for candidate, references in zip(candidate_list, references_list):
        candidate_tokens = candidate.split()
        reference_tokens = references[0].split()

        precision_scores = []
        for i in range(1, n + 1):
            cand_ng = count_ngrams(candidate_tokens, i)
            ref_ng = count_ngrams(reference_tokens, i)

            overlap = sum((cand_ng & ref_ng).values())
            total = max(sum(cand_ng.values()), 1)  # avoid division by zero
            precision_scores.append(overlap / total)

        score = np.exp(
            np.mean([np.log(p + 1e-8) for p in precision_scores])
        )  # geometric mean
        scores.append(score)

    return float(np.mean(scores))


# Load models
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    cache_dir=HUGGINGFACE_CACHE_DIR,
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# Apply LoRA to T5 decoder
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
# model = get_peft_model(model, lora_config)


class TikZGenModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, input_ids, attention_mask, image_grid_thw):
        # Inference: Generation of the output
        generated_ids = self.model.generate(
            pixel_values=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1280 * 4,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text


# Load full dataset (or first 15K examples)
full_data = load_dataset("nllg/datikz-v3", split="train[:150]").shuffle(
    seed=42
)  # todo full

# Split 90% train / 10% test
split = full_data.train_test_split(test_size=0.1, seed=42)
train_raw = split["train"]
test_raw = split["test"]


preprocessed_train = []
preprocessed_test = []

default_message = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "Using the image and its caption below, provide tikz code that would generate this image. Image caption: {}",
            },
        ],
    }
]


def preprocess_dataset(dataset_split, messages):
    processed = []
    for example in tqdm(dataset_split):
        # breakpoint()
        messages_use = deepcopy(messages)
        for id, content in enumerate(messages_use[0]["content"]):
            if "image" in content:
                messages_use[0]["content"][id]["image"] = example["image"]
            if "text" in content:
                messages_use[0]["content"][id]["text"] = messages_use[0]["content"][id][
                    "text"
                ].format(example["caption"])

        # Preparation for inference
        text = processor.apply_chat_template(
            messages_use, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_use)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # breakpoint()
        processed.append(
            {
                "image": inputs["pixel_values"],
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
                # "labels": targets["input_ids"].squeeze(0),
                "caption": example["caption"],
                "reference": example["code"],
                "pil": example["image"],
            }
        )
    return processed


preprocessed_train = preprocess_dataset(train_raw, default_message)
preprocessed_test = preprocess_dataset(test_raw, default_message)


from torch.utils.data import Dataset


class TikzDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


dataset_torch = TikzDataset(preprocessed_train)


def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "image_grid_thw": torch.stack([x["image_grid_thw"] for x in batch]),
        # "labels": torch.stack([x["labels"] for x in batch]),
        "caption": [x["caption"] for x in batch],
        "reference": [x["reference"] for x in batch],
        "pil": [x["pil"] for x in batch],
    }


loader = DataLoader(dataset_torch, batch_size=4, shuffle=True, collate_fn=collate_fn)


sample = dataset_torch[0]
print(type(sample["image"]), sample["image"].shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TikZGenModel(model).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loader = DataLoader(dataset_torch, batch_size=4, shuffle=True, collate_fn=collate_fn)

# model.train()
# for epoch in range(3):
#     loop = tqdm(loader, desc=f"Epoch {epoch+1}")
#     for batch in loop:
#         image = batch["image"].to(device)
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         image_grid_thw = batch["image_grid_thw"].to(device)

#         # labels = batch["labels"].to(device)

#         outputs = model(image, input_ids, attention_mask, image_grid_thw)  # , labels)
#         loss = outputs.loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loop.set_postfix(loss=loss.item())

#     #    torch.save(model.state_dict(), f"/home/mila/d/daria.yasafova/scratch/tikz_project/iangola/tikzgen_epoch{epoch+1}.pt")
#     torch.save(model.state_dict(), f"{save_dir}/tikzgen_epoch{epoch+1}.pt")


clip_score_metric = TorchCLIPScore(
    model_name_or_path="openai/clip-vit-base-patch32"
).to(device)

def compile_tikz_to_image(tikz_code: str, dpi: int = 300) -> Image.Image:
    """
    If `tex_content` begins with \documentclass, treat it as a full document.
    Otherwise, wrap it in a minimal standalone TikZ document.
    Compile to PDF with pdflatex, convert the first page to a PIL Image.
    """
    # Decide whether we're given a full .tex or just a snippet
    if tex_content.lstrip().startswith(r"\documentclass"):
        full_tex = tex_content
    else:
        # wrap snippet in standalone template
        tpl = (
            "\\documentclass[tikz,border=2pt]{{standalone}}\n"
            "\\usepackage{{tikz}}\n"
            "\\begin{{document}}\n"
            "{code}\n"
            "\\end{{document}}"
        )
        full_tex = tpl.format(code=tex_content)

    with tempfile.TemporaryDirectory() as tmp:
        # write out .tex
        tex_path = os.path.join(tmp, "compiled.tex")
        with open(tex_path, "w") as f:
            f.write(full_tex)

        # run pdflatex
        subprocess.run(
            ["pdflatex", "-interaction=batchmode", "-halt-on-error", "compiled.tex"],
            cwd=tmp,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # convert PDF → PIL Image
        pdf_path = os.path.join(tmp, "compiled.pdf")
        pages = convert_from_path(pdf_path, dpi=dpi)
        return pages[0]

def generate_tikz():
    """
    Modified function to handle three types of inputs:
    1. Image only
    2. Text only
    3. Both Image and Text together
    """
    model.eval()

    modes = ["image_only", "text_only", "image_and_text"]
    folder = "/home/mila/d/daria.yasafova/scratch/tikz_project/predictions"
    type = "zero-shot"
    folder_save = os.path.join(folder, type)
    for mode in modes:
        mode_save_folder = os.path.join(folder_save, mode)
        os.makedirs(mode_save_folder, exist_ok=True)
        # save the image, tikz code, and model output
        if mode == "image_only":
            message = deepcopy(default_message)
            message[0]["content"][1][
                "text"
            ] = "Provide a tikz code that creates the image"
        elif mode == "text_only":
            message = deepcopy(default_message)
            message[0]["content"] = [default_message[0]["content"][1]]

        elif mode == "image_and_text":
            message = deepcopy(default_message)
        else:
            raise Exception(f"`mode` was not provided in the right format.")
        test_data = preprocess_dataset(test_raw, message)

        loader = DataLoader(
            test_data, batch_size=4, shuffle=False, collate_fn=collate_fn
        )

        loop = tqdm(loader, desc=f"Predicting for mode {mode}")
        count = 0
        with torch.no_grad():
            for batch in loop:
                image = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)

                # labels = batch["labels"].to(device)

                outputs = model(
                    image, input_ids, attention_mask, image_grid_thw
                )  # , labels)
                for img, target_tikz, model_tikz in zip(
                    batch["pil"], batch["reference"], outputs
                ):
                    with open(
                        os.path.join(mode_save_folder, f"{count}_target.txt"), "w+"
                    ) as f:
                        f.write(target_tikz)
                    with open(
                        os.path.join(mode_save_folder, f"{count}_prediction.txt"), "w+"
                    ) as f:
                        f.write(model_tikz)

                    img.save(os.path.join(mode_save_folder, f"{count}_img.jpg"))

                    # --- NEW: compile and save the predicted TikZ as PNG ---
                    try
                        pred_img = compile_tikz_to_image(model_tikz)
                        pred_img.save(os.path.join(mode_save_folder, f"{count}_prediction.png"))
                    except Exception as e:
                        print(f"[Warning] could not compile prediction {count}: {e}")

                    count += 1


generate_tikz()

# # Sample 20 examples from your dataset
# # test_data = preprocessed_test # todo full
# test_data = preprocessed_test[:20]

# # Collect predictions and inputs
# predictions, references, images, captions = [], [], [], []

# for ex in test_data:
#     # pred = generate_tikz(image_tensor = ex["image"])
#     pred = generate_tikz(ex["caption"], ex["image"])
#     predictions.append(pred)
#     references.append(ex["reference"])  # or ex['reference'] if using preprocessed_data
#     images.append(ex["image"])
#     captions.append(ex["caption"])

# breakpoint()
# print(predictions[0])

# to_PIL = transforms.ToPILImage()


def compute_clip_score_direct(images, texts):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    # Define preprocessing transform that matches CLIP's requirements
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # Process each image
    all_image_features = []
    for img in images:
        # Apply preprocessing directly
        if isinstance(img, Image.Image):
            # Convert PIL image to tensor
            img_tensor = preprocess(img).unsqueeze(0).to(device)
        else:
            # raise TypeError(f"Expected PIL image, got {type(img)}")
            pil_img = to_PIL(img)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # Get image features
        with torch.no_grad():
            image_features = model.clip_model.get_image_features(img_tensor)
            all_image_features.append(image_features)

    # Stack all image features
    image_features = torch.cat(all_image_features, dim=0)

    # Process text inputs
    text_inputs = clip_processor.tokenizer(
        texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).to(device)

    # Get text features
    with torch.no_grad():
        text_features = model.clip_model.get_text_features(
            input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask
        )

    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity scores
    similarity = torch.sum(image_features * text_features, dim=-1)

    # Scale scores to 0-100 range (standard for CLIP score)
    scores = ((similarity + 1) * 50).cpu().numpy()

    return scores.mean().item()


# Replace the call to compute_clip_score with this
clip = compute_clip_score_direct(images, captions)


def compute_clip_score_basic(images, texts):
    """A simpler CLIP score implementation that focuses on compatibility"""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    # Initialize lists to store features
    image_features_list = []
    text_features_list = []

    # Process each image-text pair independently
    for img, text in zip(images, texts):
        try:
            # Process image
            if isinstance(img, Image.Image):
                # Make sure image is RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Use the processor to prepare the image
                image_inputs = clip_processor(images=img, return_tensors="pt").to(
                    device
                )

                # Extract features
                with torch.no_grad():
                    image_features = model.clip_model.vision_model(
                        **image_inputs
                    ).pooler_output
                    image_features_list.append(image_features)

            # Process text
            text_inputs = clip_processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            ).to(device)

            # Extract features
            with torch.no_grad():
                text_features = model.clip_model.text_model(**text_inputs).pooler_output
                text_features_list.append(text_features)

        except Exception as e:
            print(f"Error processing pair: {e}")
            # Continue with other pairs

    # If we have any successful pairs
    if image_features_list and text_features_list:
        # Stack features
        image_features = torch.cat(image_features_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity
        similarity = torch.sum(image_features * text_features, dim=-1)

        # Scale to 0-100
        scores = ((similarity + 1) * 50).cpu().numpy()

        return scores.mean().item()
    else:
        raise ValueError("Could not compute features for any image-text pair")


def compute_clip_score_hf():
    """Compute CLIP score using a fresh CLIP model from HuggingFace"""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel

    # Load pre-trained CLIP model
    pretrained_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
        device
    )
    pretrained_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize lists to store scores
    scores = []
    processed_pairs = 0
    breakpoint()
    # Process each image-text pair
    for i, (img, text) in enumerate(zip(images, captions)):
        try:
            # Always truncate the text to ensure it fits within CLIP's max length
            # Prepare inputs
            img = (img - img.min()) / (img.max() - img.min())
            inputs = pretrained_processor(
                text=[text],
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,  # Ensure truncation is enabled
                max_length=77,  # Set max length explicitly
            ).to(device)

            # Get features
            with torch.no_grad():
                outputs = pretrained_clip(**inputs)

                # Get similarity score (cosine similarity)
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.item()

                # Most CLIP scores range from 0-40 in practice
                scores.append(score)
                processed_pairs += 1

        except Exception as e:
            print(f"Error with pair {i}: {e}")

    print(f"Successfully processed {processed_pairs} out of {len(images)} pairs")

    # Return average score
    if scores:
        avg_score = sum(scores) / len(scores)
        # CLIP scores are typically between 0-40, so no need to normalize
        return avg_score
    else:
        raise ValueError("Could not compute any valid CLIP scores")


##########

clip = compute_clip_score_hf()
print("CLIPScore:", clip)

# Also print CrystalBLEU score
cb = crystal_bleu(predictions, [[ref] for ref in references])  # todo check
print("CrystalBLEU:", cb)

##########
# Use this function
"""
try:
    clip = compute_clip_score_hf()
    print("CLIPScore:", clip)

    # Also print CrystalBLEU score
    cb = crystal_bleu(predictions, [[ref] for ref in references])  # todo check
    print("CrystalBLEU:", cb)
except Exception as e:
    print(f"Could not compute CLIP score: {e}")
    # Fallback to just reporting CrystalBLEU
    print("CrystalBLEU:", cb)
"""


# Baseline scores from Belouadi et al., 2024
baselines = {
    "CrystalBLEU (Claude 2, text)": 2.80,
    "CLIPScore (Claude 2, text)": 27.10,
    "CrystalBLEU (Claude 3, sketch)": 0.61,
    "CLIPScore (Claude 3, sketch)": 26.50,
    "CrystalBLEU (SOTA)": 3.469,
    "CLIPScore (SOTA)": 29.115,
}


import pandas as pd

# Combine metrics into a DataFrame
results = {
    "Model": ["Your Model", "Claude 2", "Claude 3", "SOTA"],
    "Task": ["Text-to-TikZ", "Text-to-TikZ", "Sketch-to-TikZ", "SOTA Reference"],
    "CrystalBLEU": [round(cb, 3), 2.80, 0.61, 3.469],
    "CLIPScore": [round(clip, 3), 27.10, 26.50, 29.115],
}

df_eval = pd.DataFrame(results)

# Save to CSV
df_eval.to_csv(f"{save_dir}/tikz_eval_summary.csv", index=False)

print("✅ Saved evaluation summary to tikz_eval_summary.csv")
df_eval

breakpoint()


#####################


import evaluate

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Prepare inputs for evaluate
results_bleu = bleu.compute(
    predictions=predictions, references=[[ref] for ref in references]
)
results_rouge = rouge.compute(predictions=predictions, references=references)

# Compute Exact Match
exact_matches = [
    pred.strip() == ref.strip() for pred, ref in zip(predictions, references)
]
exact_match_score = sum(exact_matches) / len(exact_matches)

# Print results
print("📊 Evaluation Results:")
print(f"🔵 BLEU: {results_bleu['bleu']:.4f}")
print(f"🟢 ROUGE-L: {results_rouge['rougeL']:.4f}")
print(f"🟣 Exact Match: {exact_match_score:.4f}")


####################


#####################


from kid_score import compute_kid
from PIL import Image
import os

# Paths to real and generated image folders
real_dir = "real_imgs"
gen_dir = "gen_imgs"

# Generate predictions and render them as PNG images into these folders
# Then call:
kid_score = compute_kid([real_dir, gen_dir])
print("KID Score:", kid_score)


######################

breakpoint()


# Save everything together
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "clip_model_state_dict": clip_model.state_dict(),
        "t5_model_state_dict": t5_model.state_dict(),  # includes LoRA
    },
    f"{save_dir}/tikzgen_model.pt",
)
