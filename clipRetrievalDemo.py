# %%
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
from clip_retrieval.clip_client import ClipClient, Modality
import matplotlib.pyplot as plt

from IPython.display import Image, display
import urllib.request
# from PIL import Image

def log_result(result):
    id, caption, url, similarity = result["id"], result["caption"], result["url"], result["similarity"]
    print(f"id: {id}")
    print(f"caption: {caption}")
    print(f"url: {url}")
    print(f"similarity: {similarity}")
    # urllib.request.urlretrieve(url, "output/"+caption+"-sim-"+str(similarity))
    display(Image(url=url, unconfined=True))

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Torch is using " + device)

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# %% Load data
imgPath = [".assets/image/car_image.jpg", ]
audPath = [".assets/sound/car_audio.wav", ]
txtList=["A dog.", "A car", "A bird"]
# depPath = ["res/office_depth.jpg", ]

inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(imgPath, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audPath, device),
    ModalityType.TEXT: data.load_and_transform_text(txtList, device),
    # ModalityType.DEPTH: data.load_and_transform_vision_data(depPath, device)
}

with torch.no_grad():
    embeddings = model(inputs)

# plt.plot(embeddings[ModalityType.VISION][0].cpu())
# plt.plot(embeddings[ModalityType.AUDIO][0].cpu())
# plt.show()

# %% Embeddings arithmetic
# plt.plot(embeddings[ModalityType.VISION][0].cpu())
# plt.plot(embeddings[ModalityType.AUDIO][0].cpu())
# plt.show()
embVision = embeddings[ModalityType.VISION] / embeddings[ModalityType.VISION].norm(p=2, dim=-1, keepdim=True)
embAudio = embeddings[ModalityType.AUDIO] / embeddings[ModalityType.AUDIO].norm(p=2, dim=-1, keepdim=True)
embText = embeddings[ModalityType.TEXT] / embeddings[ModalityType.TEXT].norm(p=2, dim=-1, keepdim=True)
# plt.plot(embVision[0].cpu())
# plt.plot(embAudio[0].cpu())
# plt.show()

embeddingsSUM = (embVision[0] + embAudio[0])/2
# plt.plot(embeddingsSUM.cpu())
# plt.show()

# %% Query the image server
# Online CLIP images retrieval
client = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-H-14",
    num_images=10,
)
# results = client.query(embedding_input=embeddingsSUM.tolist())
results = client.query(embedding_input=embAudio[0].tolist())
for i in results:
    log_result(i)

# %%