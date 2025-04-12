
import os
import json
from pathlib import Path
from dotenv import dotenv_values
import requests
import json
from openai import OpenAI
import re

def load_env():
    """
    Загружает переменные из .env файла и возвращает их как словарь.
    """
    config = dotenv_values(".env")
    # При желании можно добавить валидацию наличия нужных ключей
    return {
        "API_KEY": config.get("OPENROUTER_API_KEY"),
        "API_BASE": config.get("OPENROUTER_API_BASE"),
        "MODEL_NAME": config.get("MODEL_NAME"),
    }


def chunk_dialogue(dialogue, chunk_size=4):
    """
    Делит список 'dialogue' на чанки по 'chunk_size' элементов (реплик).
    Возвращает список списков.
    """
    for i in range(0, len(dialogue), chunk_size):
        yield dialogue[i:i+chunk_size]

def build_input_from_chunk(chunk):
    """
    На основе 4 подряд идущих реплик строим строку вида:
    Вопрос (или реплика интервьюера): ...
    Ответ (или реплика кандидата): ...
    """
    lines = []
    for seg in chunk:
        if seg["speaker"] == "interviewer":
            lines.append(f"Вопрос: {seg['text']}")
        else:
            lines.append(f"Ответ: {seg['text']}")
    return "\n".join(lines)

def parse_json(json_output):
    """
    Убираем markdown-обрамление, если оно есть, и пробуем распарсить строку как JSON.
    """
    try:
        lines = json_output.splitlines()
        # Ищем начало блока ```json
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                # Берём всё после этой строки до следующего ```
                json_output = "".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
        # Удаляем лишние запятые перед закрывающей фигурной скобкой
        json_output = re.sub(r',\s*}', '}', json_output)

        return json.loads(json_output)
    except Exception:
        return None

def call_openrouter_api(client, prompt: str, model: str = "google/gemma-3-4b-it:free", max_tokens: int = 2048) -> str:
    """
    Формирует запрос к OpenRouter API, используя SDK OpenAI, и возвращает сгенерированный ответ.
    """
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>"
            },
            extra_body={},
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — виртуальный наставник, который анализирует диалог кандидата и интервьюера. "
                        "На основании диалога определи сильные и слабые стороны кандидата и сформулируй "
                        "персонализированные рекомендации. Исправляй грамматические ошибки и неточности в названиях. "
                        "Если роли неправильно назначены, то твоя задача понять из контекста кто является кандидатом "
                        "сформулировать для него персонализированные рекомендации. "
                        "Важно: верни строго валидный JSON, содержащий ровно одно поле 'output'. "
                        "Внутри 'output' укажи:\n"
                        "- hard_skills: [список строк]\n"
                        "- soft_skills: [список строк]\n"
                        "- recommendations: [список строк]\n"
                        "Не добавляй никаких других полей и не используй Markdown-блоки. "
                        "Пример корректной структуры:\n"
                        "{\n"
                        "  \"output\": {\n"
                        "    \"hard_skills\": [\"...\"],\n"
                        "    \"soft_skills\": [\"...\"],\n"
                        "    \"recommendations\": [\"...\"]\n"
                        "  }\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating content: {e}")
        return None


def main():
    config = load_env()
    api_key = config["API_KEY"]
    api_base = config["API_BASE"]
    model_name = config["MODEL_NAME"]

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    global_instruction = (
        "Представь, что ты виртуальный наставник. Определи сильные и слабые стороны "
        "кандидата по диалогу и сформулируй рекомендации."
    )

    prepared_data_dir = Path("prepared_data")
    output_dir = Path("finetuning_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(prepared_data_dir.glob("*_final.json"))

    for json_file in json_files:
        print(f"Обработка файла: {json_file.name}")
        
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        dialogue = data.get("dialogue", [])
        if not dialogue:
            print("Пустой диалог, пропускаем...")
            continue
        
        out_file = output_dir / f"{json_file.stem}_samples.jsonl"
        
        with out_file.open("w", encoding="utf-8") as outfile:
            for chunk_id, chunk in enumerate(chunk_dialogue(dialogue, chunk_size=4)):
                # Сформируем input
                prompt_text = build_input_from_chunk(chunk)

                # Запрос к модели
                completion = call_openrouter_api(
                    client=client, 
                    prompt=prompt_text, 
                    model=model_name, 
                    max_tokens=2048
                )
                
                if completion:
                    parsed_json = parse_json(completion)
                    
                    # Проверяем, что есть "output"
                    if parsed_json and "output" in parsed_json:
                        # Формируем итоговый JSON
                        final_data = {
                            "instruction": global_instruction,  # своя инструкция
                            "input": prompt_text,               # сам диалог
                            "output": parsed_json["output"],    # результат от модели
                            "chunk_id": chunk_id,
                            "source_file": json_file.name
                        }
                    else:
                        # Парсинг не удался или нет поля output
                        final_data = {
                            "instruction": global_instruction,
                            "input": prompt_text,
                            "raw_response": completion,
                            "chunk_id": chunk_id,
                            "source_file": json_file.name
                        }
                    
                    outfile.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                else:
                    print("Не удалось получить ответ от API. Пропускаем этот чанк.")
        print(f"Результаты сохранены в {out_file}")


if __name__ == "__main__":
    main()