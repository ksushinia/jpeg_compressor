from PIL import Image
import os

# Пути к изображениям
images = {
    "Lenna": "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/Lenna.png",
    "Lake": "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/lake.jpg"
}

# Папка для сохранения результатов
output_folder = "C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/converted/"
os.makedirs(output_folder, exist_ok=True)

def convert_and_save(name, path):
    img = Image.open(path)

    # 1. Оттенки серого
    gray = img.convert("L")
    gray.save(os.path.join(output_folder, f"{name}_gray.png"))

    # 2. Чёрно-белое без дизеринга (порог 128)
    bw_no_dither = gray.point(lambda x: 255 if x > 128 else 0, mode='1')
    bw_no_dither.save(os.path.join(output_folder, f"{name}_bw_nodither.png"))

    # 3. Чёрно-белое с дизерингом (Floyd–Steinberg)
    bw_dither = gray.convert("1")  # по умолчанию используется дизеринг
    bw_dither.save(os.path.join(output_folder, f"{name}_bw_dither.png"))

    print(f"✔ {name} обработано и сохранено.")

if __name__ == "__main__":
    for name, path in images.items():
        convert_and_save(name, path)
