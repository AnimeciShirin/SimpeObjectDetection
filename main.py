import torch
from urllib.request import urlopen
from PIL import Image
import timm
import os
import class_names

# Resim seçimi
# Yalnızca dosya adı yazabilir, ya da tüm path'i girebilirsiniz
file_name = input("Resim dosyasını giriniz: ")
image_path = os.getcwd() + "\\" + file_name if os.getcwd() not in file_name else file_name

# Belirtilen dosya bulunamazsa hata verecek
if not os.path.exists(image_path):
    print(f"\n {image_path} \n")
    raise Exception("Hata: Bu dosya mevcut değil.")

# Resmi açıyor
img = Image.open(image_path)

# Modeli tanımlıyor
model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in1k', pretrained=True)
model = model.eval()

# Modelin gerektirdiği dönüşümleri oluşturuyoruz (normalizasyon, yeniden boyutlandırma)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Resmi dönüştürüyoruz ve modele girdi olarak veriyoruz
input_tensor = transforms(img).unsqueeze(0)  # resmi bir batch boyutuna çeviriyoruz
output = model(input_tensor)

# Tahmin edilen sınıf ve olasılıkları alıyoruz
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

# Tahmin sonuçlarını ekrana yazdırıyoruz
print("\nEn yüksek ihtimalle tahmin edilen 5 cevap: \n\n")
for i in range(5):
    class_idx = top5_class_indices[0][i].item()
    probability = top5_probabilities[0][i].item()
    class_name = class_names.names[class_idx]
    print(f"Sınıf: {class_name}, Olasılık: %{probability:.2f}")