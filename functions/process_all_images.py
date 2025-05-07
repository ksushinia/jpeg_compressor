import os
import matplotlib.pyplot as plt
from PIL import Image
from compressor_with_packing import compress_to_file, decompress_from_file

INPUT_FOLDER = 'C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/input_images'
OUTPUT_FOLDER = 'C:/Users/79508/Desktop/4 семестри/АИСД/2 лабораторная/коди/images/output_images'
COMPRESSED_FOLDER = os.path.join(OUTPUT_FOLDER, 'compressed')
RESTORED_FOLDER = os.path.join(OUTPUT_FOLDER, 'restored')
GRAPHS_FOLDER = os.path.join(OUTPUT_FOLDER, 'graphs')

# Создаем все необходимые папки
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(RESTORED_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

# Уровни качества для графика (начинаем с 1, затем шаг 5)
GRAPH_QUALITIES = [1] + list(range(5, 101, 5))  # 1, 5, 10, 15,..., 100 (всего 21 точка)
# Основные уровни качества для восстановления (должны быть в GRAPH_QUALITIES)
RESTORE_QUALITIES = [1, 20, 40, 60, 80, 100]


def process_compression(image_path, quality):
    """Выполняет только сжатие и возвращает результаты"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    compressed_path = os.path.join(COMPRESSED_FOLDER, f"{base_name}_q{quality}.bin")

    # Сжатие
    compress_to_file(image_path, compressed_path, quality=quality)
    compressed_size = os.path.getsize(compressed_path)

    return {
        'quality': quality,
        'compressed_size': compressed_size,
        'compression_ratio': os.path.getsize(image_path) / compressed_size
    }


def restore_image(image_path, quality):
    """Выполняет восстановление изображения"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    compressed_path = os.path.join(COMPRESSED_FOLDER, f"{base_name}_q{quality}.bin")
    output_path = os.path.join(RESTORED_FOLDER, f"{base_name}_q{quality}.png")

    # Восстановление и сохранение в PNG
    decompress_from_file(compressed_path, output_path)

    # Убедимся, что файл в формате PNG
    img = Image.open(output_path)
    if img.format != 'PNG':
        img.save(output_path, format='PNG')

    return os.path.getsize(output_path)


def create_compression_graph(image_name, results, restore_qualities):
    """Создает и сохраняет график сжатия"""
    qualities = [r['quality'] for r in results]
    compressed_sizes = [r['compressed_size'] / 1024 for r in results]  # в KB

    plt.figure(figsize=(12, 6))
    plt.plot(qualities, compressed_sizes, 'b-o', linewidth=2, markersize=5)

    # Фильтруем точки восстановления, которые есть в результатах
    restore_points = [(q, s) for q, s in zip(qualities, compressed_sizes)
                      if q in restore_qualities]

    if restore_points:  # Проверяем, что есть точки для отображения
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
    for image_name in os.listdir(INPUT_FOLDER):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"\n=== Обработка {image_name} ===")
        input_path = os.path.join(INPUT_FOLDER, image_name)
        original_size = os.path.getsize(input_path)

        # 1. Сжатие для всех точек графика (1,5,10,...,100)
        print("Выполнение сжатия для графика (21 точка):")
        compression_results = []
        for quality in GRAPH_QUALITIES:
            res = process_compression(input_path, quality)
            compression_results.append(res)
            print(f"q{quality:3d}: {res['compressed_size'] / 1024:6.1f} KB",
                  end=' | ' if quality % 20 != 0 else '\n')

        # 2. Сразу сохраняем график
        graph_path = create_compression_graph(image_name, compression_results, RESTORE_QUALITIES)
        print(f"\nГрафик сохранен: {graph_path}")

        # 3. Восстановление для ключевых точек
        print("\nВосстановление изображений для ключевых точек качества:")
        for quality in RESTORE_QUALITIES:
            restored_size = restore_image(input_path, quality)
            print(f"q{quality:3d}: восстановлено как PNG ({restored_size / 1024:.1f} KB)")

        # 4. Вывод сводки
        print("\nСводка по ключевым точкам:")
        print(f"{'Качество':<8} | {'Сжатый (KB)':<12} | {'Сжатие (x)':<10}")
        print("-" * 35)
        for q in RESTORE_QUALITIES:
            res = next(r for r in compression_results if r['quality'] == q)
            print(f"{q:<8} | {res['compressed_size'] / 1024:<12.2f} | {res['compression_ratio']:<10.2f}")

        print(f"\nОригинальный размер: {original_size / 1024:.2f} KB")

    print("\n=== Обработка завершена ===")
    print(f"\nРезультаты сохранены в:")
    print(f"- Сжатые файлы: {COMPRESSED_FOLDER}")
    print(f"- Восстановленные PNG: {RESTORED_FOLDER}")
    print(f"- Графики: {GRAPHS_FOLDER}")


if __name__ == "__main__":
    main()