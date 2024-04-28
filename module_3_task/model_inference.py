import torch
from torchvision import transforms
import cv2
from PIL import Image
from module_3_task.model import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(img_path):
  """Preprocesses an image for inference."""
  img = cv2.imread(img_path)
  if img is None:
    raise ValueError(f"Failed to load image: {img_path}")

  # Convert OpenCV BGR image to RGB (expected by torchvision)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(img)  # Convert NumPy array to PIL Image

  resize = transforms.Resize((224, 224))  # Assuming AlexNet
  normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  preprocess = transforms.Compose([resize, transforms.ToTensor(), normalize])
  img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
  return img_tensor.to(device)  # Move to same device as the model


def predict(model_path, img_path):
  """Loads the model and makes inference on an image."""
  # Load the model state dictionary
  model = ResNet18()
  model = model.to(device) 
  model.load_state_dict(torch.load(model_path))
  model.eval()

  # Preprocess the image
  img_tensor = preprocess_image(img_path)

  # Perform inference
  with torch.no_grad():
    output = model(img_tensor)

  # Get the predicted class label (assuming you have 2 classes)
  predicted_class = torch.argmax(output, dim=1).item()
  classes = ['ANIMAL_STICKERS', 'BATH_SPONGE', 'COLOURING_PENCILS', 'CUSHION_COVER', 'Pencil_Eraser', 'Pencil_Sharpener', 'RED_PURSE', 'WASHBAG']
  return classes[predicted_class]

# creating a dictionary to map the class index to the class name



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./module_3_task/my_model_weights.pth"
    img_path = './Images/ANIMAL_STICKERS/71bBdl5RLDL._AC_UL320_.jpg'
    predicted_class = predict(model_path, img_path)
    print(f"Predicted class: {predicted_class}")