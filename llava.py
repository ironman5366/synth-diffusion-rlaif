from PIL import Image
from pathlib import Path
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(os.environ["MODEL_DIR"])

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=MODEL_DIR)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=MODEL_DIR)


def label_slice(prompt: str, image_paths: list[str]) -> str:
    prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
    images = [Image.open(im) for im in image_paths]
    inputs = processor(text=prompt, images=images, return_tensors="pt")

    # Generate
    generate_ids = model.generate(**inputs, max_length=30)
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
