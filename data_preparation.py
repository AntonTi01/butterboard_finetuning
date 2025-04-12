#!/usr/bin/env python3
import json
import sys
import logging
from datetime import datetime
from docx import Document
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Список ключевых слов для интервьюера
KEYWORDS = [
    "расскажи", "представь", "объясни",
    "опиши", "назови", "уточни", 
    "прокомментируй", "приведи пример",
    "как бы", "сравни", "почему ты",
    "почему",
    "что именно", "в чем заключается",
    "какой алгоритм", "какие технологии",
    "предложи решение", "раскрой", "какой стэк",
]

def parse_timestamp(ts_str: str) -> datetime:
    """
    Преобразует строку с временной меткой (в формате "HH:MM:SS,fff" или с точкой) в объект datetime.
    00:00:03,799 -> 00:00:03,799 -> datetime.datetime(1900, 1, 1, 0, 0, 3, 799000)
    """
    ts_str = ts_str.strip().replace(".", ",")
    try:
        return datetime.strptime(ts_str, "%H:%M:%S,%f")
    except ValueError as e:
        logging.error(f"Ошибка при разборе временной метки: {ts_str} - {e}")
        raise

def parse_diarization_file(filepath: Path) -> list:
    """
    Читает файл диаризации и возвращает список сегментов.
    Каждая запись содержит:
      - timestamp_start: оригинальная строка времени начала
      - timestamp_end: оригинальная строка времени окончания
      - speaker: имя спикера
      - start_dt: объект datetime начала
      - end_dt: объект datetime окончания
    """
    segments = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" - ")
            if len(parts) != 3:
                logging.warning(f"Неверный формат строки диаризации: {line}")
                continue
            ts_start, ts_end, speaker = parts
            try:
                start_dt = parse_timestamp(ts_start)
                end_dt = parse_timestamp(ts_end)
            except ValueError:
                continue
            segments.append({
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "speaker": speaker.strip(),
                "start_dt": start_dt,
                "end_dt": end_dt
            })
    return segments

def parse_transcription_docx(filepath: Path) -> list:
    """
    Читает файл транскрипции (docx) и возвращает список сегментов.
    Каждый сегмент содержит:
      - timestamp_start: строка времени начала
      - timestamp_end: строка времени окончания
      - text: текст сегмента
      - start_dt: объект datetime начала
      - end_dt: объект datetime окончания
    """
    segments = []
    doc = Document(filepath)
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or " - " not in text:
            continue
        parts = text.split(" - ", 2)
        if len(parts) != 3:
            logging.warning(f"Неверный формат строки транскрипции: {text}")
            continue
        ts_start, ts_end, seg_text = parts
        try:
            start_dt = parse_timestamp(ts_start)
            end_dt = parse_timestamp(ts_end)
        except ValueError:
            continue
        segments.append({
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "text": seg_text.strip(),
            "start_dt": start_dt,
            "end_dt": end_dt
        })
    return segments

def calculate_overlap(seg_start, seg_end, ref_start, ref_end):
    """
    Вычисляет длительность перекрытия двух сегментов (в секундах).
    """
    latest_start = max(seg_start, ref_start)
    earliest_end = min(seg_end, ref_end)
    overlap = (earliest_end - latest_start).total_seconds()
    return max(0, overlap)

def match_segments(trans_segments: list, diar_segments: list, tolerance_ratio: float = 0.5) -> list:
    """
    Для каждого сегмента транскрипции находит лучший соответствующий сегмент диаризации
    по правилу максимального перекрытия. Сегмент считается сопоставленным,
    если длительность перекрытия >= tolerance_ratio * длительность сегмента транскрипции.
    
    Возвращает список объединённых сегментов с ключами:
      - start_time (строка)
      - end_time (строка)
      - speaker (из диаризации)
      - text (из транскрипции)
    """
    matched = []
    for t_seg in trans_segments:
        best_overlap = 0.0
        best_diar = None
        t_duration = (t_seg["end_dt"] - t_seg["start_dt"]).total_seconds()
        for d_seg in diar_segments:
            overlap = calculate_overlap(t_seg["start_dt"], t_seg["end_dt"],
                                        d_seg["start_dt"], d_seg["end_dt"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_diar = d_seg
        if best_diar and best_overlap >= tolerance_ratio * t_duration:
            matched.append({
                "start_time": t_seg["timestamp_start"],
                "end_time": t_seg["timestamp_end"],
                "speaker": best_diar["speaker"],
                "text": t_seg["text"]
            })
        else:
            logging.info(f"Сегмент транскрипции [{t_seg['timestamp_start']} - {t_seg['timestamp_end']}] не сопоставлен, недостаточное перекрытие.")
    return matched

def assign_roles(dialogue: list) -> list:
    """
    Определяет роли «interviewer» и «candidate» для топ-2 спикеров (по количеству сегментов).
    Для каждого из них подсчитывается суммарное количество вопросительных знаков и вхождений ключевых слов.
    Спикеру с большим количеством таких признаков назначается роль «interviewer».
    
    В итоговом списке остаются только сегменты, принадлежащие этим двум спикерам,
    а исходное имя спикера заменяется на соответствующую роль.
    """
    # Сгруппировать статистику по каждому спикеру
    speaker_stats = {}
    for seg in dialogue:
        spk = seg["speaker"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"count": 0, "questions": 0, "keywords": 0}
        speaker_stats[spk]["count"] += 1
        speaker_stats[spk]["questions"] += seg["text"].count("?")
        text_lower = seg["text"].lower()
        for kw in KEYWORDS:
            speaker_stats[spk]["keywords"] += text_lower.count(kw)
    
    # Отбираем топ-2 спикеров по количеству сегментов
    top_speakers = sorted(
        speaker_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:2]

    if len(top_speakers) < 2:
        logging.warning("Найдено менее двух спикеров в диалоге.")
        return dialogue  # Если спикеров меньше, оставляем без назначения ролей
    
    spk1, spk2 = top_speakers[0][0], top_speakers[1][0]
    
    # Подсчёт итоговых баллов: (вопросы + ключевые слова)
    score1 = speaker_stats[spk1]["questions"] + speaker_stats[spk1]["keywords"]
    score2 = speaker_stats[spk2]["questions"] + speaker_stats[spk2]["keywords"]
    
    logging.info(
        f"Top-1 speaker: {spk1}, segments={speaker_stats[spk1]['count']}, "
        f"questions={speaker_stats[spk1]['questions']}, keywords={speaker_stats[spk1]['keywords']}, "
        f"total_score={score1}"
    )
    logging.info(
        f"Top-2 speaker: {spk2}, segments={speaker_stats[spk2]['count']}, "
        f"questions={speaker_stats[spk2]['questions']}, keywords={speaker_stats[spk2]['keywords']}, "
        f"total_score={score2}"
    )

    # Назначаем роли
    if score1 >= score2:
        role_mapping = {spk1: "interviewer", spk2: "candidate"}
    else:
        role_mapping = {spk2: "interviewer", spk1: "candidate"}
    
    logging.info(f"Роли назначены: {role_mapping}")
    
    # Обновляем диалог: оставляем только сегменты для top-2 спикеров и заменяем их имена на роли
    updated_dialogue = []
    for seg in dialogue:
        if seg["speaker"] in role_mapping:
            new_seg = seg.copy()
            new_seg["speaker"] = role_mapping[new_seg["speaker"]]
            updated_dialogue.append(new_seg)
    # Сортируем по началу сегмента (используем парсинг временной метки)
    updated_dialogue.sort(key=lambda x: parse_timestamp(x["start_time"]))
    return updated_dialogue


def merge_consecutive_segments(dialogue: list) -> list:
    """
    Объединяет подряд идущие сегменты с одинаковой ролью.
    Для объединения:
      - start_time берется из первого сегмента,
      - end_time из последнего сегмента последовательной группы,
      - текст объединяется через пробел.
    """
    if not dialogue:
        return []
    
    merged = []
    current = dialogue[0].copy()
    
    for seg in dialogue[1:]:
        if seg["speaker"] == current["speaker"]:
            # Обновляем конечное время и объединяем текст
            current["end_time"] = seg["end_time"]
            current["text"] = current["text"].strip() + " " + seg["text"].strip()
        else:
            merged.append(current)
            current = seg.copy()
    merged.append(current)
    return merged

def process_pair(diar_file: Path, trans_file: Path, prepared_data_dir: Path):
    """
    Обрабатывает пару файлов: diar_file (из папки diarization) и trans_file (из папки transcription).
    Выполняет parse, match, assign_roles, merge_consecutive_segments и сохраняет результат.
    """
    logging.info(f"Обрабатываем пару:\n  diarization={diar_file}\n  transcription={trans_file}")

    # Получаем "базовое имя" для сохранения
    # Например, "interview_syst_analyst_1"
    base_name = diar_file.name.split("_diarization")[0]  # либо более точный split
    # Или, если нужно быть точными, можно взять "interview_syst_analyst_1" в trans_file тоже

    # Читаем сегменты диаризации
    diar_segments = parse_diarization_file(diar_file)
    logging.info(f"[{base_name}] Найдено {len(diar_segments)} сегментов диаризации.")

    # Читаем сегменты транскрипции
    trans_segments = parse_transcription_docx(trans_file)
    logging.info(f"[{base_name}] Найдено {len(trans_segments)} сегментов транскрипции.")

    # Сопоставляем
    dialogue = match_segments(trans_segments, diar_segments, tolerance_ratio=0.5)
    logging.info(f"[{base_name}] Сопоставлено {len(dialogue)} сегментов диалога.")

    # Назначаем роли
    final_dialogue = assign_roles(dialogue)

    # Сохраняем не объединённый вариант
    final_json = {"dialogue": final_dialogue}
    out_unmerged = prepared_data_dir / f"dialoge_{base_name}.json"
    with out_unmerged.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    logging.info(f"[{base_name}] Не объединённый JSON сохранён в {out_unmerged}")

    # Объединяем подряд идущие сегменты
    merged_dialogue = merge_consecutive_segments(final_dialogue)
    merged_json = {"dialogue": merged_dialogue}
    out_merged = prepared_data_dir / f"dialoge_{base_name}_final.json"
    with out_merged.open("w", encoding="utf-8") as f:
        json.dump(merged_json, f, ensure_ascii=False, indent=2)
    logging.info(f"[{base_name}] Объединённый JSON сохранён в {out_merged}")


def main():
    """
    Если передано 2 аргумента (diar_file, trans_file), обрабатываем только их.
    Иначе берём все файлы *_diarization_processed_final.txt в папке diarization и
    ищем соответствующие *_cleaned_final.docx в папке transcription, сопоставляя их по
    базовому имени (до '_diarization' и до '_cleaned').
    """
    args = sys.argv[1:]
    prepared_data_dir = Path("prepared_data")
    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    diarization_dir = Path("raw_data/diarization")
    transcription_dir = Path("raw_data/transcribation")

    if len(args) == 2:
        # Запущен с указанием конкретных файлов
        diar_file = Path(args[0])
        trans_file = Path(args[1])
        process_pair(diar_file, trans_file, prepared_data_dir)
    else:
        # Собираем все файлы из diarization и transcription
        diar_files = list(diarization_dir.glob("*_diarization_processed_final.txt"))
        trans_files = list(transcription_dir.glob("*_cleaned_final.docx"))

        # Построим словари: {base_name -> Path}
        # например "interview_syst_analyst_1" -> Path(".../interview_syst_analyst_1_diarization_processed_final.txt")
        diar_map = {}
        for df in diar_files:
            base = df.name.split("_diarization")[0]  # interview_syst_analyst_1
            diar_map[base] = df

        trans_map = {}
        for tf in trans_files:
            base = tf.name.split("_cleaned")[0]  # interview_syst_analyst_1
            trans_map[base] = tf

        # Пытаемся найти пары по совпадающему base
        common_bases = set(diar_map.keys()) & set(trans_map.keys())
        if not common_bases:
            logging.warning("Не найдено пар файлов с совпадающим базовым именем!")
            return

        for base in common_bases:
            dfile = diar_map[base]
            tfile = trans_map[base]
            process_pair(dfile, tfile, prepared_data_dir)


if __name__ == "__main__":
    main()