from PIL import Image
import requests
from pathlib import Path
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(os.environ["MODEL_DIR"])

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=MODEL_DIR)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=MODEL_DIR)

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
