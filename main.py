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

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tempfile import NamedTemporaryFile


MODEL_FOLDER = Path("/Users/will/Projects/krea/video_pipeline/data/models/")

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=MODEL_FOLDER
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="cpu",
    trust_remote_code=True,
    cache_dir=MODEL_FOLDER,
).eval()
model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=MODEL_FOLDER
)


def label_slice(prompt: str, image_paths: list[str]) -> str:
    # TODO: this should be batched at some pint
    query = tokenizer.from_list_format(
        [{"image": im} for im in image_paths] + [{"text": prompt}]
    )
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


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

for batch in loader:
    # TODO: batch this later
    for it in batch:
        prompt = f"Which of these images has better aesthetic quality, and best fits the prompt, '{caption}'."
        "Output '1' if the first image is better, '2' if the "
        "second image is better. Only output '1' or '2', do not output anything else"
        print(f"Comparing images {it['image_0_path']} and {it['image_1_path']} with {prompt}")
        caption = it["caption"]
        response = label_slice(
            prompt,
            image_paths=[it["image_0_path"], it["image_1_path"]],
        )

        if it["image_0_uid"] == it["best_image_uid"]:
            better_image = "1"
        else:
            better_image = "2"

        assert response in {"1", "2"}, "AI Bad :("

        if response == better_image:
            agree_with_human += 1

        total += 1

        print(it)
