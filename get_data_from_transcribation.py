import shutil
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def move_final_files(source_dir: Path, destination_dir: Path, extension: str):
    """
    Перемещает файлы из source_dir в destination_dir, если имя файла заканчивается на '_final'
    и файл имеет заданное расширение.
    
    Параметры:
      source_dir (Path): исходная директория, где находятся файлы.
      destination_dir (Path): целевая директория для перемещения.
      extension (str): расширение файла, например ".docx" или ".txt".
    """
    if not source_dir.is_dir():
        logging.error(f"Исходная директория {source_dir} не существует.")
        return
    
    # Создаём целевую директорию, если её нет
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем файлы по паттерну
    pattern = f"*{extension}"
    files = list(source_dir.glob(pattern))
    if not files:
        logging.info(f"В директории {source_dir} не найдено файлов с расширением {extension}.")
    
    # Перебор найденных файлов
    for file in files:
        # Проверяем, что имя файла (без расширения) заканчивается на '_final'
        if file.stem.endswith("_final"):
            dest_file = destination_dir / file.name
            try:
                shutil.copy2(str(file), str(dest_file))
                logging.info(f"Скопирован файл: {file} -> {dest_file}")
            except Exception as e:
                logging.error(f"Ошибка при перемещении {file}: {e}")


def main():
    # Задаём пути исходных директорий
    trans_source = Path("/Users/forcemajor01/data_science/work_place/butterboard/butterboard_interview_analysis/results/postprocessed")
    diar_source = Path("/Users/forcemajor01/data_science/work_place/butterboard/butterboard_interview_analysis/results/diarization")
    
    # Задаём пути целевых директорий
    trans_destination = Path("/Users/forcemajor01/data_science/work_place/butterboard/finetuning/raw_data/transcribation")
    diar_destination = Path("/Users/forcemajor01/data_science/work_place/butterboard/finetuning/raw_data/diarization")
    
    # Перемещаем файлы для транскрибации (.docx)
    move_final_files(trans_source, trans_destination, ".docx")
    
    # Перемещаем файлы для диаризации (.txt)
    move_final_files(diar_source, diar_destination, ".txt")


if __name__ == "__main__":
    main()
