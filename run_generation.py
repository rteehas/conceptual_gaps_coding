from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from functools import partial
import re
from argparse import ArgumentParser
import numpy as np


device = "cuda"

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M').to(device)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")


def get_completion(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length = 1000, do_sample=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

def split_on_whitespace(string):
    # Split the string on whitespace using regular expression
    parts = re.split(r'(\s+)', string)
    
    # Initialize lists to store the split parts and whitespace characters
    split_parts = []
    whitespace_chars = []
    
    # Iterate over the split parts
    for part in parts:
        if re.match(r'\s+', part):
            # If the part is a whitespace character, add it to whitespace_chars
            whitespace_chars.append(part)
        else:
            # If the part is not a whitespace character, add it to split_parts
            split_parts.append(part)
    
    return split_parts, whitespace_chars


def truncate_dataset(ex, proportion = 0.5):
    text = ex['text']
    parts, chars = split_on_whitespace(text)

    trunc_length = int(len(parts) * proportion)
    to_join = []
    for p, c in zip(parts[:trunc_length - 1], chars):
        to_join.append(p + c)

    to_join.append(parts[trunc_length - 1])
    ex['truncated text'] = "".join(to_join)
    return ex

def add_completions(ex):
    for i in range(10):
        ex["completion_{}".format(i)] = get_completion(ex['truncated text'])
    # ex['completion'] = 
    return ex
    
    
truncate_20 = partial(truncate_dataset, proportion=0.2)
truncate_40 = partial(truncate_dataset, proportion=0.4)
truncate_60 = partial(truncate_dataset, proportion=0.6)

values = [20, 40, 60]
truncations = [truncate_20, truncate_40, truncate_60]

dataset = load_dataset("roneneldan/TinyStories")
for i, val in enumerate(values):
    truncate_function = truncations[i]
    sample_size = 100
    random_sample = np.random.choice(list(range(len(dataset['train']))), size = sample_size, replace=False)

    tiny_data = dataset['train'].select(random_sample.tolist()).map(truncate_function)
    completed = tiny_data.map(add_completions)
    completed.save_to_disk("/scratch/rst306/tinystories_outputs/model_33M/train/completions_truncated_{}".format(val))


# tiny_20 = 