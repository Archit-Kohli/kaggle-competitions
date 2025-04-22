from dataclasses import dataclass
from pathlib import Path
import modal

app = modal.App(name="example-dreambooth-flux")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.31.0",
    "datasets~=2.14.6",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "gradio~=5.5.0",
    "huggingface-hub==0.26.2",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft==0.11.1",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "starlette==0.41.2",
    "transformers~=4.41.2",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "wandb==0.17.6",
)

GIT_SHA = (
    "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch
)

image = (
    image.apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)

@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    instance_name: str = "v3ct0r"
    # class_name: str = "Golden Retriever"
    model_name: str = "black-forest-labs/FLUX.1-schnell"

volume = modal.Volume.from_name(
    "dreambooth-finetuning-volume-flux", create_if_missing=True
)

MODEL_DIR = "/model"
DATASET_PATH = "ArchitKohli/vector-lora-dataset"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # turn on faster downloads from HF
)

@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,  # 10 minutes
)
def download_models(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)

USE_WANDB = True

@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 2
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117


@app.function(
    image=image,
    gpu="A100-80GB",  # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1800,  # 30 minutes
    secrets=[huggingface_secret]
    + (
        [
            modal.Secret.from_name(
                "wandb-secret", required_keys=["WANDB_API_KEY"]
            )
        ]
        if USE_WANDB
        else []
    ),
)
def train(config):
    import subprocess
    from accelerate.utils import write_basic_config

    write_basic_config(mixed_precision="bf16")

    caption_column = "text"
    image_column = "image"
    prompt = "A v3ct0r image of"

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={MODEL_DIR}",
            f"--dataset_name={DATASET_PATH}",
            f"--caption_column={caption_column}",
            f"--image_column={image_column}",
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            f"--push_to_hub"
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )
    volume.commit()


@app.cls(image=image, gpu="A100", volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()

        # set up a hugging face inference pipeline using our model
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @modal.method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""
    num_inference_steps: int = 50
    guidance_scale: float = 8


web_image = image.add_local_dir(
    Path(__file__).parent / "assets",
    remote_path="/assets",
)
@app.function(
    image=web_image,
    max_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()

    instance_phrase = f"a v3ct0r image of"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    modal_docs_url = "https://modal.com/docs"
    modal_example_url = f"{modal_docs_url}/examples/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

### Learn how to make a "Dreambooth" for your own pet [here]({modal_example_url}).
    """

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(
        theme=theme,
        css=css,
        title=f"Generate images of {config.instance_name} on Modal",
    ) as interface:
        gr.Markdown(
            f"# Generate images of {instance_phrase}.\n\n{description}",
        )
        with gr.Row():
            inp = gr.Textbox(  # input text component
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                lines=10,
            )
            out = gr.Image(  # output image component
                height=512, width=512, label="", min_width=512, elem_id="output"
            )
        with gr.Row():
            btn = gr.Button("Dream", variant="primary", scale=2)
            btn.click(
                fn=go, inputs=inp, outputs=out
            )  # connect inputs and outputs with inference function

            gr.Button(  # shameless plug
                "⚡️ Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

# - `modal run dreambooth_app.py` will train the model. Change the `instance_example_urls_file` to point to your own pet's images.
# - `modal serve dreambooth_app.py` will [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the Gradio interface at a temporary location. Great for iterating on code!
# - `modal shell dreambooth_app.py` is a convenient helper to open a bash [shell](https://modal.com/docs/guide/developing-debugging#interactive-shell) in our image. Great for debugging environment issues.

@app.local_entrypoint()
def run(
    max_train_steps: int = 1500,
):
    print("🎨 loading model")
    download_models.remote(SharedConfig())
    print("🎨 setting up training")
    config = TrainConfig(max_train_steps=max_train_steps)
    train.remote(config)
    print("🎨 training finished")