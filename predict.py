# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        local_files_only = True  # set to true if models are cached
        cache_dir = "model_cache"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm-chat-hf",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            .to("cuda")
            .eval()
        )

    def predict(
        self,
        query: str = Input(description="Input query.", default="Describe this image."),
        image: Path = Input(description="Input image."),
        vqa: bool = Input(default=False, description="Enable vqa mode."),
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        if vqa:
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=[],
                images=[image],
                template_version="vqa",
            )
        else:
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer, query=query, history=[], images=[image]
            )

        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(outputs[0]).strip()
        if response.endswith("</s>"):
            response = response[: -len("</s>")]
        return response
