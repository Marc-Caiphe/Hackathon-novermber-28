# doc_generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

def generate_documentation(code):
    inputs = tokenizer(f"Document this code:\n\n{code}", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate code documentation")
    parser.add_argument("file", help="Path to the code file")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        code = f.read()

    documentation = generate_documentation(code)
    print("\nGenerated Documentation:\n")
    print(documentation)