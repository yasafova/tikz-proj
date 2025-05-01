"""this is a modified copy of paraphrase.py"""

""" todo rename, todo github"""

import pandas as pd

# from google import genai  # ensure you have the gemini client library installed
import os, re
from tqdm import tqdm

# import time
# import sys
import numpy as np

# from huggingface_hub import InferenceClient
# breakpoint()
from using_unsloth import make_inference
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--start", type=int, required=True, help="Start value of the dataset for processing"
)
parser.add_argument(
    "--end", type=int, required=True, help="End value of the dataset for processing"
)

args = parser.parse_args()

START = args.start
END = args.end

tqdm.pandas()

# HOME_DIR = os.getenv('HOME')
SCRATCH_DIR = os.getenv("SCRATCH")
HUGGINGFACE_CACHE_DIR = os.path.join(SCRATCH_DIR, "huggingface")
TIKZ_DIR = os.path.join(SCRATCH_DIR, "tikz_project")


def get_db_password(file_path="keys.txt", type="DB_PASSWORD"):
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{type}="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        # try to get it from os
        return os.getenv(type)
    raise ValueError(f"`{type}`not found in the file.")


# # Initialize Gemini client
# gemini_api_key = get_db_password(type='GEMINI_API_KEY')
# client = genai.Client(api_key=gemini_api_key)


# hf_client = InferenceClient(
#     provider="hf-inference",
#     api_key="hf_***"
# )


def paraphrase_with_unsloth(review_text, folder, count):

    filename = os.path.join(folder, f"{count}_textdescr.txt")
    if not os.path.exists(filename):

        paraphrased_text = make_inference(review_text)
        # breakpoint()
        pattern = r"(?s).*#Output_description:\s*(.*)$"
        match = re.search(pattern, paraphrased_text[0])
        if match:
            paraphrased_text = match.group(1).replace("<|eot_id|>", "")
            # This regex pattern matches everything from <|start_header_id|> to <|end_header_id|> (non-greedily)
            pattern = (
                r"(?:<\|start_header_id\|>|<\|end_header_id\|>).*?<\|end_header_id\|>"
            )

            # The re.DOTALL flag ensures the dot matches newline characters too
            paraphrased_text = re.sub(
                pattern, "", paraphrased_text, flags=re.DOTALL
            ).strip()
            if paraphrased_text != "":
                with open(filename, "w+") as f:
                    f.write(paraphrased_text)


def main(df, folder, inner_split_use):
    # Ensure that the DataFrame contains the expected columns.
    if "Review" not in df.columns:
        raise ValueError("The CSV file must contain a 'Review' column.")

    for ind in tqdm(inner_split_use, desc="Paraphrasing..."):
        row = df.loc[ind]
        paraphrase_with_unsloth(row["Review"], folder, ind)


def load_datikz(start, end):
    # Load full dataset (or first n examples)
    full_data = load_dataset(
        "nllg/datikz-v3",
        split=f"train[{start}:{end}]",
        cache_dir=HUGGINGFACE_CACHE_DIR,
    )
    datikz_new = full_data.select_columns(["code", "image"])
    return datikz_new


if __name__ == "__main__":

    ##############
    # breakpoint()
    start = START
    end = END
    data = load_datikz(start, end)
    folder = os.path.join(TIKZ_DIR, "datasets/synthetic_data")

    # Extract and save each code element
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
        idx_new = idx + start
        # print(idx_new)
        code = item["code"]
        # print(code)
        # paraphrase_with_unsloth(code, folder, idx)
        paraphrase_with_unsloth(
            code.replace("{", "{{").replace("}", "}}"), folder, idx_new
        )
        # with open(
        #    f"{folder}/{idx_new}_code.txt",
        #    "w",
        # ) as output_file:
        #    output_file.write(code)

    #################
