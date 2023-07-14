# %%
import anything2image.imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from IPython.display import Image, display

# %% construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)
model = ib.imagebind_huge(pretrained=True).eval().to(device)

# %% generate image
with torch.no_grad():
    embeddings = model.forward({
        ib.ModalityType.VISION: ib.load_and_transform_vision_data([".assets/image/dog_image.jpg"], device),
    })
    img_embeddings = embeddings[ib.ModalityType.VISION]
    # img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)

    embeddings = model.forward({
        ib.ModalityType.AUDIO: ib.load_and_transform_audio_data([".assets/sound/car_audio.wav"], device),
    }, normalize=False)
    audio_embeddings = embeddings[ib.ModalityType.AUDIO]
    # audio_embeddings = audio_embeddings / audio_embeddings.norm(p=2, dim=-1, keepdim=True)

    embeddingsCombo = (img_embeddings + audio_embeddings)/2

# %% generate image
with torch.no_grad():
    images = pipe(image_embeds=embeddingsCombo.half()).images
    # images[0].save("audio2img.png")
for image in images:
    display(image)
# %%
