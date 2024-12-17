
import os
import torch
from torchvision import transforms, models
from PIL import Image
import argparse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ResNet18 model
MODEL = models.resnet18(pretrained=True)
MODEL.eval()
MODEL = MODEL.to(device)

# Image Preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to Process Image
def process_image(image_path):
    if not os.path.isfile(image_path):
        return "Ошибка: Файл не найден."
    if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10 MB
        return "Ошибка: Размер файла превышает 10 МБ."
    try:
        image = Image.open(image_path).convert("RGB")
        return TRANSFORM(image).unsqueeze(0).to(device)
    except Exception as e:
        return f"Ошибка обработки изображения: {e}"

# Classify Image Function
def classify_image(image_tensor):
    categories = ['Кошки', 'Собаки', 'Автомобили', 'Природа']
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        _, predicted = outputs.max(1)
        return categories[predicted.item() % len(categories)]

def main():
    parser = argparse.ArgumentParser(description="Классификация изображений с использованием ResNet18")
    parser.add_argument("image_path", type=str, help="Путь к изображению (.jpg, .jpeg, .png)")
    args = parser.parse_args()

    image_tensor = process_image(args.image_path)
    if isinstance(image_tensor, str):
        print(image_tensor)
    else:
        category = classify_image(image_tensor)
        print(f"Изображение относится к категории: {category}")

if __name__ == "__main__":
    main()
