from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=MODEL_FOLDER
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="cuda",
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
