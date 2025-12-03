import vk_api
import json
import time
import re

# --- КОНФИГУРАЦИЯ ---
SERVICE_TOKEN = 'ВАШ_СЕРВИСНЫЙ_КЛЮЧ_ДОСТУПА'  # Вставьте сюда ваш токен
GROUP_DOMAIN = 'habr'  # Короткое имя группы (например, vk.com/habr -> habr)
POSTS_COUNT = 500      # Сколько постов скачать (для теста начните с малого)
OUTPUT_FILE = 'vk_dataset.json'

def clean_text(text):
    """
    Базовая очистка текста для RAG.
    Убирает ссылки на профили вида [id123|Name] и лишние пробелы.
    """
    if not text:
        return ""
    # Убираем конструкции [club123|Название] или [id123|Имя]
    text = re.sub(r'\[(club|id)\d+\|(.+?)\]', r'\2', text)
    return text.strip()

def parse_vk_wall(domain, count, token):
    # Авторизация
    vk_session = vk_api.VkApi(token=token)
    vk = vk_session.get_api()

    posts_data = []
    offset = 0
    step = 100  # Максимум за один запрос API отдает 100 постов

    print(f"🚀 Начинаем сбор {count} постов из сообщества '{domain}'...")

    while offset < count:
        try:
            # Если нужно скачать меньше 100, берем остаток
            count_to_get = min(step, count - offset)

            response = vk.wall.get(domain=domain, count=count_to_get, offset=offset)
            items = response['items']

            if not items:
                break

            for post in items:
                # Пропускаем рекламу (marked_as_ads)
                if post.get('marked_as_ads', 0) == 1:
                    continue

                raw_text = post.get('text', '')
                
                # Пропускаем посты без текста (например, только фото)
                if not raw_text:
                    continue

                # Формируем объект данных для RAG
                doc = {
                    'id': post['id'],
                    'date': post['date'], # Unix timestamp
                    'text': clean_text(raw_text),
                    'likes': post['likes']['count'],
                    'views': post.get('views', {}).get('count', 0),
                    'url': f"https://vk.com/{domain}?w=wall{post['owner_id']}_{post['id']}"
                }
                posts_data.append(doc)

            offset += step
            print(f"✅ Обработано {len(posts_data)} постов...")
            
            # Пауза, чтобы не превысить лимиты API (3 запроса в секунду)
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            break

    return posts_data

if __name__ == "__main__":
    data = parse_vk_wall(GROUP_DOMAIN, POSTS_COUNT, SERVICE_TOKEN)
    
    # Сохраняем в JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Готово! Сохранено {len(data)} записей в '{OUTPUT_FILE}'")
