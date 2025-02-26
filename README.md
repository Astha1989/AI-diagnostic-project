# MAIRA-2 Chest X-Ray Analysis

This repository contains a Python project that utilizes the Microsoft MAIRA-2 model for analyzing chest X-ray images. The model is loaded using the Hugging Face `transformers` library and processes medical images.

## Features
- Uses **MAIRA-2**, a model from Microsoft, for medical image analysis.
- Downloads sample chest X-ray images from the IU-Xray dataset.
- Processes images using **Hugging Face Transformers**.

## Installation

To set up the project, install the required dependencies:

```sh
pip install git+https://github.com/huggingface/transformers.git@88d960937c81a32bfb63356a2e8ecf7999619681 gradio
pip install torch requests pillow huggingface_hub
```

## Usage

Run the Jupyter Notebook to load the model, process sample images, and analyze results.

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Install dependencies (as shown above).
3. Open and run `maira_2.ipynb` in Jupyter Notebook.

## Example Code

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

# Download and open an image
url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
image = Image.open(requests.get(url, stream=True).raw)

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs)
```

## License
This project uses data from IU-Xray, which is licensed under CC. The MAIRA-2 model is provided by Microsoft.

## Acknowledgments
- [Microsoft MAIRA-2](https://huggingface.co/microsoft/maira-2)
- [IU-Xray Dataset](https://openi.nlm.nih.gov/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

