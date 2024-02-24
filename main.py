"""
Basic idea:
- Download pick-a-pic
- Evaluate aesthetic score with different models and different prompts
- See what method aligns best with humans
- Throw that into DRLX
"""

from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from accelerate import Accelerator
from tempfile import NamedTemporaryFile

#dataset = load_dataset("yuvalkirstain/pickapic_v2")
dataset = load_dataset("kashif/pickascore", split="validation")

accelerator = Accelerator()

def collate_fn(examples):
    cleaned_data = []

    for example in examples:
        del example["created_at"]
        for image_bytes, key_name in [(example["jpg_0"], "image_0_path"), (example["jpg_1"], "image_1_path")]:
            with NamedTemporaryFile(mode="wb", suffix=".jpg", delete=False) as f:
                f.write(image_bytes)
                example[key_name] = f.name
        cleaned_data.append(example)

    return cleaned_data

loader = DataLoader(dataset, batch_size=1,
                    collate_fn=collate_fn,)

agree_with_human = 0
total = 0

constraint_suffix =("Output '1' if the first image is better, '2' if the second image is better. "
                    "Only output '1' or '2', do not output anything else")

def base_caption(caption: str):
    return f"Which of these images is best described by the caption '{caption}'? {constraint_suffix}"

def base_no_caption(_caption: str):
    return f"Which of these images has better aesthetic quality?. {constraint_suffix}"

prompt_options = {
    "base_caption": base_caption,
    "base_no_caption": base_no_caption
}

prompt_accuracy = defaultdict(int)

for batch in tqdm(loader):
    # TODO: batch this later
    for it in batch:
        for prompt_key, prompt_fn in prompt_options.items():
            caption = it["caption"]
            prompt = prompt_fn(caption)

            print(f"Comparing images {it['image_0_path']} and {it['image_1_path']} with {prompt}")

            response = label_slice(
                prompt,
                image_paths=[it["image_0_path"], it["image_1_path"]],
            )

            if it["image_0_uid"] == it["best_image_uid"]:
                better_image = "1"
            else:
                better_image = "2"

            print(response)
            assert response in {"1", "2"}, "AI Bad :("

            if response == better_image:
                prompt_accuracy[prompt_key] += 1

            print(f"AI response: {response}, human response: {better_image}, agree: {response == better_image}")

        total += 1

for prompt_key, accuracy in prompt_accuracy.items():
    print(f"Prompt {prompt_key} accuracy: {accuracy} / {total}, {accuracy / total}")