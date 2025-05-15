import os
import matplotlib.pyplot as plt
from PIL import Image
from compressor_with_packing import compress_to_file, decompress_from_file

INPUT_FOLDER = "C:/Users/Elizaveta/OneDrive/Документы/jpeg_compressor/images/input_images"
OUTPUT_FOLDER = "C:/Users/Elizaveta/OneDrive/Документы/jpeg_compressor/images/output_images"
COMPRESSED_FOLDER = os.path.join(OUTPUT_FOLDER, 'compressed')
RESTORED_FOLDER = os.path.join(OUTPUT_FOLDER, 'restored')
GRAPHS_FOLDER = os.path.join(OUTPUT_FOLDER, 'graphs')

# Создаем все необходимые папки
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(RESTORED_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

# Основные уровни качества для восстановления изображений
RESTORE_QUALITIES = [1, 20, 40, 60, 80, 100]
# Все уровни качества для графика (включая дополнительные точки)
GRAPH_QUALITIES = sorted(set(RESTORE_QUALITIES + list(range(0, 101, 5))))


def process_image(image_path, quality, restore=False):
    """Обрабатывает изображение с заданным качеством"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    compressed_path = os.path.join(COMPRESSED_FOLDER, f"{base_name}_q{quality}.bin")

    # Сжатие
    compress_to_file(image_path, compressed_path, quality=quality)
    compressed_size = os.path.getsize(compressed_path)

    result = {
        'quality': quality,
        'compressed_size': compressed_size,
        'compression_ratio': os.path.getsize(image_path) / compressed_size
    }

    # Восстановление только если требуется
    if restore:
        output_path = os.path.join(RESTORED_FOLDER, f"{base_name}_q{quality}.png")
        decompress_from_file(compressed_path, output_path)

        # Конвертируем в PNG
        img = Image.open(output_path)
        if img.format != 'PNG':
            img.save(output_path, format='PNG')

        result['restored_size'] = os.path.getsize(output_path)

    return result


def create_compression_graph(image_name, results):
    """Создает и сохраняет график сжатия"""
    qualities = [r['quality'] for r in results]
    compressed_sizes = [r['compressed_size'] / 1024 for r in results]  # в KB

    plt.figure(figsize=(12, 6))

    # Основной график
    plt.plot(qualities, compressed_sizes, 'b-', linewidth=2)

    # Точки восстановления выделяем красным
    restore_points = [(q, s) for q, s in zip(qualities, compressed_sizes)
                      if q in RESTORE_QUALITIES]
    if restore_points:
        restore_q, restore_s = zip(*restore_points)
        plt.scatter(restore_q, restore_s, c='red', s=100,
                    label='Точки восстановления', zorder=5)

    plt.title(f'Зависимость размера сжатого файла от качества\n{image_name}')
    plt.xlabel('Качество сжатия')
    plt.ylabel('Размер сжатого файла (KB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    graph_path = os.path.join(GRAPHS_FOLDER, f"{os.path.splitext(image_name)[0]}_compression.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()

    return graph_path


def main():
    # 1. Сначала обрабатываем ключевые точки для восстановления
    print("=== Этап 1: Восстановление изображений ===")
    restore_results = {}

    for image_name in os.listdir(INPUT_FOLDER):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"\nОбработка {image_name} для восстановления:")
        input_path = os.path.join(INPUT_FOLDER, image_name)
        image_results = []

        for quality in RESTORE_QUALITIES:
            try:
                res = process_image(input_path, quality, restore=True)
                image_results.append(res)
                if 'restored_size' in res:
                    print(
                        f"q{quality:3d}: сжато {res['compressed_size'] / 1024:.1f} KB, восстановлено {res['restored_size'] / 1024:.1f} KB")
                else:
                    print(f"q{quality:3d}: ошибка восстановления")
            except Exception as e:
                print(f"q{quality:3d}: ошибка обработки - {str(e)}")
                image_results.append({
                    'quality': quality,
                    'compressed_size': 0,
                    'compression_ratio': 0
                })

        restore_results[image_name] = image_results

    # 2. Затем обрабатываем дополнительные точки для графиков
    print("\n=== Этап 2: Подготовка данных для графиков ===")
    graph_results = {}

    for image_name in os.listdir(INPUT_FOLDER):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"\nДополнительная обработка {image_name} для графиков:")
        input_path = os.path.join(INPUT_FOLDER, image_name)

        # Получаем уже обработанные точки
        existing_results = restore_results.get(image_name, [])
        existing_qualities = {r['quality'] for r in existing_results}

        image_results = existing_results.copy()

        # Обрабатываем только новые точки качества
        for quality in GRAPH_QUALITIES:
            if quality in existing_qualities:
                continue

            try:
                res = process_image(input_path, quality, restore=False)
                image_results.append(res)
                print(f"q{quality:3d}: сжато {res['compressed_size'] / 1024:.1f} KB")
            except Exception as e:
                print(f"q{quality:3d}: ошибка сжатия - {str(e)}")
                image_results.append({
                    'quality': quality,
                    'compressed_size': 0,
                    'compression_ratio': 0
                })

        graph_results[image_name] = sorted(image_results, key=lambda x: x['quality'])

    # 3. Строим графики для всех изображений
    print("\n=== Этап 3: Построение графиков ===")
    for image_name, results in graph_results.items():
        try:
            graph_path = create_compression_graph(image_name, results)
            print(f"График для {image_name} сохранен: {graph_path}")
        except Exception as e:
            print(f"Ошибка построения графика для {image_name}: {str(e)}")

    print("\n=== Обработка завершена ===")
    print(f"\nРезультаты сохранены в:")
    print(f"- Сжатые файлы: {COMPRESSED_FOLDER}")
    print(f"- Восстановленные PNG: {RESTORED_FOLDER}")
    print(f"- Графики: {GRAPHS_FOLDER}")


if __name__ == "__main__":
    main()