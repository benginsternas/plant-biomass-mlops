import torch
import mlflow.pyfunc
from PIL import Image
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BiomassPyFuncWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = torch.load(
            context.artifacts["torch_model"],
            map_location="cpu",
            weights_only=False,
        )
        self.model.eval()
        with open(context.artifacts["max_weight_val"]) as f:
            self.max_weight_val = float(f.read())

    def predict(self, context, model_input):
        tensors = []
        for img in model_input:
            img = img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            tensors.append(_TRANSFORM(img))

        batch = torch.stack(tensors)
        with torch.no_grad():
            preds = self.model(batch)
        return (preds.squeeze(1) * self.max_weight_val).tolist()
