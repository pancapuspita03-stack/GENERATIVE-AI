# 12.4 AutoModel - Loading Models Directly (Halaman 139)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
)
model.eval()

prompt = "The key advantage of local LLMs is"
inputs = tokenizer(prompt, return_tensors="pt")

if __name__ == "__main__":
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated)
