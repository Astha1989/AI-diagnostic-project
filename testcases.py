import unittest
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
import torch

def download_image(url: str) -> Image.Image:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(response.raw)

class TestMairaModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
        cls.processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
        cls.sample_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"

    def test_model_loading(self):
        self.assertIsInstance(self.model, AutoModelForCausalLM)

    def test_processor_loading(self):
        self.assertIsInstance(self.processor, AutoProcessor)

    def test_image_download(self):
        image = download_image(self.sample_url)
        self.assertIsInstance(image, Image.Image)

    def test_model_processing(self):
        image = download_image(self.sample_url)
        inputs = self.processor(images=image, return_tensors="pt")
        self.assertIn("pixel_values", inputs)
        self.assertIsInstance(inputs["pixel_values"], torch.Tensor)

if __name__ == "__main__":
    unittest.main()
