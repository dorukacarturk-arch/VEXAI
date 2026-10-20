import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def reply(message, history):
    messages = [{"role": "system", "content": "Sen VEXAI adlı yardımcı bir yapay zekâsın. Türkçe, açık ve samimi cevap ver."}]
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.extend([
                {"role": "user", "content": str(item[0])},
                {"role": "assistant", "content": str(item[1])}
            ])
    messages.append({"role": "user", "content": message})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()

css = """
body {background:#080b0a!important}
.gradio-container {max-width:1000px!important;margin:auto!important}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), css=css, title="VEXAI") as demo:
    gr.Markdown("# 🟢 VEXAI\n### Yapay zekâ asistanın")
    gr.ChatInterface(
        fn=reply,
        chatbot=gr.Chatbot(height=520),
        textbox=gr.Textbox(placeholder="VEXAI'ye mesaj yaz...", container=False),
        title="",
        description="Sorunu yaz, birlikte çözelim."
    )

demo.launch()
