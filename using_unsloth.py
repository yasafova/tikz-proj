from unsloth import FastLanguageModel
import torch

# max_seq_length = 4096*4096 # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir="/home/mila/d/daria.yasafova/scratch/huggingface",
    max_seq_length=10000,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    max_seq_length=10000,
)


# use this for the larger models
tikz_prompt = """
[INST] 

    You are given a TikZ diagram. Your task is to write a precise, step-by-step textual description of the diagram, focusing on all visual and structural elements necessary to reconstruct the original TikZ code as closely as possible.

    Follow these instructions:

        List All Diagram Elements:

            Identify and describe every graphical element (nodes, shapes, lines, arrows, curves, labels, etc.).

            For each element, specify its type (e.g., circle, rectangle, arrow, node, label).

        Specify Positions and Coordinates:

            Provide exact coordinates for each element if available, or describe relative positions (e.g., "node A is at (0,0)", "node B is 2 units to the right of node A").

            If grid or axes are present, mention their scale and orientation.

        Detail Styles and Attributes:

            Describe colors, line styles (solid, dashed, dotted), thickness, fill, opacity, and arrowhead types.

            Mention any custom styles or TikZ options used (e.g., draw=blue, thick, fill=red!20).

        Describe Node Content and Shape:

            For each node, specify its text/content, shape (circle, rectangle, etc.), and any additional formatting (font, size, bold, italic).

            Note if nodes are labeled or named for reference in edges.

        Explain Connections and Edges:

            For each edge or connection, specify the start and end nodes, edge style (straight, curved, bent), arrow direction, and any labels on the edges.

        Order of Drawing:

            If relevant, note the drawing order (e.g., background shapes before foreground nodes) to ensure correct layering.

        Additional Features:

            Mention any decorations, patterns, grids, legends, or annotations present in the diagram.

        Be Unambiguous and Concise:

            Use clear, technical language. Avoid subjective or interpretive descriptions.

            Your description should enable a model to reproduce the diagram in TikZ as closely as possible to the original code.

        Describe all text elements and labels:

    For every piece of text (including words, letters, numbers, or symbols), specify its exact content.

    Indicate the precise position of each text element:

        If the text is associated with a diagram element (such as a node, edge, or shape), state clearly which element it is attached to or written on.

        If the text is positioned independently (not attached to any element), provide its coordinates or describe its location relative to other elements.

    For labels on edges or arrows, specify which edge or arrow they are labeling, and indicate the label’s position (e.g., "midway along the arrow from node A to node B" or "above the edge").

    Include any relevant formatting details, such as font size, style (bold, italic), or color, if specified in the diagram.


Example TikZ code:
\\begin{{tikzpicture}}
\\node[draw, circle] (A) at (0,0) {{Start}};
\\node[draw, rectangle] (B) at (2,0) {{Process}};
\\draw[->] (A) -- (B);
\\end{{tikzpicture}}

Expected Output description of the Example TikZ code:
"A flowchart with two nodes: circular 'Start' node connected by an arrow to rectangular 'Process' node, aligned horizontally". [/INST]

Now your task:
#TikZ_code: {}
#Output_description:

"""


EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference\


def make_inference(prompt):
    # prompt = prompt  + EOS_TOKEN

    # FastLanguageModel.for_inference(model)  # Enable native 2x faster inference\

    inputs = tokenizer(
        [
            tikz_prompt.format(prompt)
            + EOS_TOKEN
            # alpaca_prompt.format(
            #     "Paraphrase the input text. Preserve the original meaning but use different wording. Put the paraphrased text in the response section", # instruction
            #     prompt, # input
            #     "", # output - leave this blank for generation!
            # ) + EOS_TOKEN
            # f"Paraphrase the input text. Preserve the original meaning but use different wording. Return only the paraphrased version: \n {prompt}"+EOS_TOKEN
        ],
        return_tensors="pt",
    ).to("cuda")
    # breakpoint()

    # inputs = tokenizer(
    # [
    #     prompt
    # ], return_tensors = "pt").to("cuda")
    # print(f'Forward pass...')
    outputs = model.generate(**inputs, use_cache=True, max_new_tokens=512)
    # from transformers import TextStreamer
    # text_streamer = TextStreamer(tokenizer)
    # outputs = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
    # print(f'Forward pass...DONE')

    output_text = tokenizer.batch_decode(outputs)
    # print(output_text)
    return output_text
