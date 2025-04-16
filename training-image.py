# ✅ Clean install of all required packages
#!pip install  datasets transformers peft accelerate bitsandbytes torchmetrics

# 🧠 Step 2: Import modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    CLIPProcessor,
    CLIPModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.multimodal import CLIPScore as TorchCLIPScore


save_dir = "/home/mila/d/daria.yasafova/scratch/tikz_project/iangola"


from collections import Counter
from nltk.util import ngrams
import numpy as np


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
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Apply LoRA to T5 decoder
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
t5_model = get_peft_model(t5_model, lora_config)


class TikZGenModel(nn.Module):
    def __init__(self, clip_model, t5_model):
        super().__init__()
        self.clip_model = clip_model
        self.t5_model = t5_model
        # breakpoint()
        IMAGE_FEATURE_DIM = clip_model.config.projection_dim
        TEXT_FEATURE_DIM = t5_model.config.d_model
        self.fusion = nn.Linear(IMAGE_FEATURE_DIM, TEXT_FEATURE_DIM)  # image only mode

        # self.fusion = nn.Linear(
        #    clip_model.config.projection_dim + t5_model.config.d_model,
        #   t5_model.config.d_model,
        # )

    def forward(self, image=None, input_ids=None, attention_mask=None, labels=None):
        image_features = None
        text_features = None

        if image is not None:
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=image)

        if input_ids is not None and attention_mask is not None:
            text_features = self.t5_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]

        if image_features is not None and text_features is not None:
            combined_features = torch.cat([text_features, image_features], dim=1)
        elif image_features is not None:
            combined_features = image_features
        elif text_features is not None:
            combined_features = text_features
        else:
            raise ValueError("At least one of image or text input must be provided.")

        fused = self.fusion(combined_features).unsqueeze(1)
        return self.t5_model(inputs_embeds=fused, labels=labels)


# Load full dataset (or first 15K examples)
full_data = load_dataset("nllg/datikz-v3", split="train[:15000]").shuffle(
    seed=42
)  # todo full

# Split 90% train / 10% test
split = full_data.train_test_split(test_size=0.1, seed=42)
train_raw = split["train"]
test_raw = split["test"]


preprocessed_train = []
preprocessed_test = []


def preprocess_dataset(dataset_split):
    processed = []
    for example in dataset_split:
        image = example["image"].convert("RGB")
        image_tensor = clip_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        inputs = tokenizer(
            example["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        targets = tokenizer(
            example["code"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        processed.append(
            {
                "image": image_tensor,
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": targets["input_ids"].squeeze(0),
                "caption": example["caption"],
                "reference": example["code"],
            }
        )
    return processed


preprocessed_train = preprocess_dataset(train_raw)
preprocessed_test = preprocess_dataset(test_raw)


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
        "input_ids": None,  # torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": None,  # torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "caption": [x["caption"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


loader = DataLoader(dataset_torch, batch_size=4, shuffle=True, collate_fn=collate_fn)


sample = dataset_torch[0]
print(type(sample["image"]), sample["image"].shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TikZGenModel(clip_model, t5_model).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loader = DataLoader(dataset_torch, batch_size=4, shuffle=True, collate_fn=collate_fn)

model.train()
for epoch in range(3):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        image = batch["image"].to(device)
        labels = batch["labels"].to(device)
        # Set text inputs to None
        outputs = model(image=image, input_ids=None, attention_mask=None, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    #    torch.save(model.state_dict(), f"/home/mila/d/daria.yasafova/scratch/tikz_project/iangola/tikzgen_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), f"{save_dir}/tikzgen_epoch{epoch+1}.pt")


clip_score_metric = TorchCLIPScore(
    model_name_or_path="openai/clip-vit-base-patch32"
).to(device)


def generate_tikz(caption=None, image_tensor=None):
    """
    Modified function to handle three types of inputs:
    1. Image only
    2. Text only
    3. Both Image and Text together
    """
    # breakpoint()
    model.eval()

    if caption is not None and image_tensor is not None:
        # Multimodal input: Image and Text together
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        input_ids = tokenizer(caption, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(caption, return_tensors="pt").attention_mask.to(
            device
        )

        with torch.no_grad():
            text_features = model.t5_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            image_features = model.clip_model.get_image_features(
                pixel_values=image_tensor
            )
            combined = model.fusion(
                torch.cat([text_features, image_features], dim=1)
            ).unsqueeze(1)
            generated_ids = model.t5_model.generate(
                inputs_embeds=combined, max_length=256, num_beams=4, early_stopping=True
            )

    elif caption is not None:
        # Text only input
        input_ids = tokenizer(caption, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(caption, return_tensors="pt").attention_mask.to(
            device
        )

        with torch.no_grad():
            text_features = model.t5_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            fused = text_features.unsqueeze(1)
            generated_ids = model.t5_model.generate(
                inputs_embeds=fused, max_length=256, num_beams=4, early_stopping=True
            )

    elif image_tensor is not None:
        # Image only input
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            image_features = model.clip_model.get_image_features(
                pixel_values=image_tensor
            )
            fused = model.fusion(image_features).unsqueeze(1)
            generated_ids = model.t5_model.generate(
                inputs_embeds=fused, max_length=256, num_beams=4, early_stopping=True
            )

    else:
        raise ValueError(
            "At least one of `caption` or `image_tensor` must be provided."
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# Sample 20 examples from your dataset
# test_data = preprocessed_test # todo full
test_data = preprocessed_test[:20]

# Collect predictions and inputs
predictions, references, images, captions = [], [], [], []

for ex in test_data:
    pred = generate_tikz(image_tensor=ex["image"])  # image only mode
    # pred = generate_tikz(ex["caption"], ex["image"])
    predictions.append(pred)
    references.append(ex["reference"])  # or ex['reference'] if using preprocessed_data
    images.append(ex["image"])
    captions.append(ex["caption"])

breakpoint()
print(predictions[0])

to_PIL = transforms.ToPILImage()


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
