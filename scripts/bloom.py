import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import modules.scripts as scripts
import gradio as gr
from modules.processing import Processed, process_images

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bloom-560m-finetuned-sd-prompts' 

tokenizer = BloomTokenizerFast.from_pretrained(ckpt)
model = BloomForCausalLM.from_pretrained(ckpt).to(device)

def generate_prompt(text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, repetition_penalty=1.05, max_length=2048, eos_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(output[0], skip_special_tokens=False).replace('</s>','')


class Script(scripts.Script):

    def title(self):
        return "sd-bloom560"

    def ui(self, is_img2img):
        with gr.Row():
            base_prompt = gr.Textbox(label='Base prompt:', lines=2)
            fetch_bloom = gr.Button('Bloom It!')
        with gr.Row():
            out_prompt = gr.Textbox(label='Bloomed Prompt', lines=3)    
        
        fetch_bloom.click(
            fn=generate_prompt,
            inputs=[
                base_prompt,
            ],
            outputs=[
                out_prompt,
            ]
            )

        return [base_prompt, fetch_bloom, out_prompt]

    def run(self, p, base_prompt, fetch_bloom, out_prompt):
        images = []
        p.prompt = out_prompt
        proc = process_images(p)
        images += proc.images

        return Processed(p, images, p.seed, proc.info)
