import time
import re
import sys
import json
from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI

##################################################
WAIT_SECONDS = 2.0  # задержка между запросами (в секундах)
MAX_RETRIES = 2     # сколько раз пытаться повторить запрос при неудаче
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Вопрос: Расскажи, пожалуйста, как ты описываешь функциональные и нефункциональные требования "
            "на проекте, и какие сложности обычно возникают?\n"
            "Ответ: В основном я пишу спецификации для функциональных требований, но нефункциональные требования "
            "тоже затрагивал. При этом описывал SLA, нагрузочные параметры и требования к отказоустойчивости. "
            "Основная сложность — это согласование с командой разработки и стейкхолдерами."
        )
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            "  \"output\": {\n"
            "    \"hard_skills\": [\"функциональные требования\", \"описание нефункциональных требований\"],\n"
            "    \"soft_skills\": [\"умение работать в команде\", \"коммуникабельность\"],\n"
            "    \"recommendations\": [\n"
            "      \"Углубить знания в области нефункциональных требований, чтобы лучше понимать, как они влияют на качество системы.\",\n"
            "      \"Развивать навыки документирования, чтобы более четко и ясно описывать требования.\",\n"
            "      \"Работать над формулировкой своих мыслей, чтобы избежать неясностей в общении.\"\n"
            "    ]\n"
            "  }\n"
            "}"
        )
    }
]

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
        # Формируем список сообщений, где:
        #   1) идёт общее системное указание,
        #   2) добавляются few-shot-примеры,
        #   3) потом идёт пользовательский контент prompt.
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — виртуальный наставник, который анализирует диалог кандидата на позицию системного аналитика и интервьюера. "
                    "На основании диалога определи сильные и слабые стороны кандидата и сформулируй "
                    "персонализированные рекомендации. Исправляй грамматические ошибки и неточности в названиях. "
                    "Если роли неправильно назначены, то твоя задача понять из контекста кто является кандидатом "
                    "и сформулировать для него персонализированные рекомендации. "
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
            }
        ]

        # Добавляем few-shot примеры
        messages.extend(FEW_SHOT_EXAMPLES)

        # Добавляем текущий пользовательский запрос
        messages.append({
            "role": "user",
            "content": prompt
        })

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>"
            },
            extra_body={},
            model=model,
            messages=messages,
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
    client = OpenAI(base_url=api_base, api_key=api_key)

    # Проверяем, передан ли путь к конкретному файлу:
    if len(sys.argv) > 1:
        file_to_process = Path(sys.argv[1])
        if not file_to_process.is_file():
            print(f"Файл {file_to_process} не найден.")
            return
        # Если файл найден - обрабатываем только его
        files = [file_to_process]
    else:
        # Иначе берём все файлы из prepared_data_dir
        prepared_data_dir = Path("prepared_data")
        files = list(prepared_data_dir.glob("*_final.json"))

    output_dir = Path("finetuning_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in files:
        print(f"\nОбработка файла: {json_file.name}")
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dialogue = data.get("dialogue", [])
        if not dialogue:
            print("Пустой диалог, пропускаем...")
            continue
        
        out_file = output_dir / f"{json_file.stem}_samples.jsonl"

        chunks = list(chunk_dialogue(dialogue, chunk_size=4))
        total_chunks = len(chunks)

        with out_file.open("w", encoding="utf-8") as outfile:
            for chunk_id, chunk in enumerate(chunks):
                print(f"Обрабатываем чанк {chunk_id + 1} из {total_chunks}...")
                prompt_text = build_input_from_chunk(chunk)

                # Делаем несколько попыток (retry), если ответ не получен
                response_content = None
                for attempt in range(MAX_RETRIES + 1):
                    if attempt > 0:
                        print(f"Повторная попытка #{attempt} для чанка {chunk_id}")
                    # Задержка перед запросом
                    time.sleep(WAIT_SECONDS)

                    completion = call_openrouter_api(
                        client=client,
                        prompt=prompt_text,
                        model=model_name,
                        max_tokens=2048
                    )
                    if completion:
                        response_content = completion
                        break
                    else:
                        print("Не удалось получить ответ от API.")
                        # Если еще есть попытки - будем пробовать снова
                        # Если нет, прервёмся и пропустим этот чанк

                # Если после всех попыток нет ответа, переходим к следующему чанку
                if not response_content:
                    print(f"Пропускаем чанк {chunk_id} окончательно — нет ответа.")
                    continue

                # Иначе парсим ответ
                parsed_json = parse_json(response_content)
                if parsed_json and "output" in parsed_json:
                    final_data = {
                        "instruction": (
                            "Представь, что ты виртуальный наставник. "
                            "Определи сильные и слабые стороны кандидата по диалогу и сформулируй рекомендации."
                        ),
                        "input": prompt_text,
                        "output": parsed_json["output"],
                        "chunk_id": chunk_id,
                        "source_file": json_file.name
                    }
                else:
                    final_data = {
                        "instruction": (
                            "Представь, что ты виртуальный наставник. "
                            "Определи сильные и слабые стороны кандидата по диалогу и сформулируй рекомендации."
                        ),
                        "input": prompt_text,
                        "raw_response": response_content,
                        "chunk_id": chunk_id,
                        "source_file": json_file.name
                    }

                outfile.write(json.dumps(final_data, ensure_ascii=False) + "\n")
        
        print(f"Результаты сохранены в {out_file}")

if __name__ == "__main__":
    main()