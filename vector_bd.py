import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = 'vk_dataset.json'
DB_DIRECTORY = 'vk_vector_db'  # Папка, куда сохранится база

# Модель для русского языка. 
# MiniLM - быстрая и легкая. Если нужно супер-качество, возьмите 'intfloat/multilingual-e5-large'
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_and_process_data(filepath):
    """Загружает JSON и превращает в Documents LangChain"""
    if not os.path.exists(filepath):
        print(f"❌ Файл {filepath} не найден! Сначала запустите парсер.")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        # Метаданные нужны, чтобы RAG мог сказать: "Я взял это из поста от такого-то числа (ссылка)"
        metadata = {
            "source_id": item['id'],
            "date": item['date'],
            "likes": item['likes'],
            "url": item['url']
        }
        
        # Создаем документ. page_content - это то, по чему будем искать смысл.
        doc = Document(page_content=item['text'], metadata=metadata)
        documents.append(doc)
    
    return documents

def create_vector_db(documents):
    if not documents:
        return

    print(f"🔄 Разбиваем {len(documents)} постов на чанки...")
    
    # 1. Чанкинг (Text Splitting)
    # chunk_size=1000: размер кусочка текста (в символах)
    # chunk_overlap=200: перекрытие, чтобы не терять смысл на границах разрыва
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️ Получилось {len(splits)} чанков (фрагментов).")

    print(f"🧠 Загружаем модель эмбеддингов ({MODEL_NAME})...")
    # Используем CPU, если нет CUDA. Этой модели достаточно CPU.
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    print("💾 Создаем векторную базу (это может занять время)...")
    # Создаем и сохраняем базу на диск
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_DIRECTORY
    )
    
    print(f"✅ База успешно сохранена в папку '{DB_DIRECTORY}'")
    return vector_db

def test_search(query):
    """Проверка: ищем ответ на вопрос в созданной базе"""
    print(f"\n🔎 Тестовый поиск по запросу: '{query}'")
    
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embedding_model)
    
    # Ищем 3 самых похожих куска текста
    results = db.similarity_search(query, k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Результат {i+1} (Лайков: {res.metadata['likes']}) ---")
        print(f"Текст: {res.page_content[:200]}...") # Показываем первые 200 символов
        print(f"Ссылка: {res.metadata['url']}")

if __name__ == "__main__":
    # 1. Загружаем
    docs = load_and_process_data(INPUT_FILE)
    
    # 2. Создаем базу (запустите один раз, потом можно закомментировать)
    create_vector_db(docs)
    
    # 3. Тестируем поиск
    test_search("Как войти в IT?")
