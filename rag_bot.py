import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- КОНФИГУРАЦИЯ ---
# Вставьте сюда ваш ключ от Google AI Studio
GOOGLE_API_KEY = "ВАШ_GOOGLE_API_KEY"

# Путь к базе, созданной на прошлом этапе
DB_DIRECTORY = 'vk_vector_db' 
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_rag_chain():
    """
    Собирает цепочку RAG: Эмбеддинги -> Векторная БД -> LLM
    """
    
    # 1. Загружаем ту же модель эмбеддингов, что и при создании базы
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 2. Подключаемся к существующей базе ChromaDB
    if not os.path.exists(DB_DIRECTORY):
        raise FileNotFoundError(f"База {DB_DIRECTORY} не найдена! Запустите скрипт из Шага 2.")
        
    vector_db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    
    # 3. Настраиваем "Ретривер" (поисковик)
    # k=3 означает, что мы берем 3 самых похожих куска текста из базы
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 4. Инициализируем LLM (Google Gemini 1.5 Flash - быстрая и бесплатная)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3 # Низкая температура, чтобы модель меньше фантазировала
    )
    
    # 5. Создаем промпт (инструкцию) для модели
    # Мы жестко говорим ей использовать ТОЛЬКО контекст.
    prompt_template = """
    Ты — умный помощник, обученный на постах из сообщества ВКонтакте.
    Твоя задача — ответить на вопрос пользователя, используя ИСКЛЮЧИТЕЛЬНО предоставленный ниже контекст.
    
    Если в контексте нет информации для ответа, честно скажи: "В постах сообщества нет информации об этом."
    Не придумывай факты от себя.
    
    Контекст (информация из постов):
    {context}
    
    Вопрос пользователя:
    {question}
    
    Твой ответ:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # 6. Собираем готовую цепочку (Chain)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" значит "засунуть все найденные куски в один промпт"
        retriever=retriever,
        return_source_documents=True, # Чтобы мы видели ссылки на посты
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_bot(query):
    try:
        chain = get_rag_chain()
        print(f"\n🤖 Думаю над вопросом: '{query}'...")
        
        # Запуск цепи
        result = chain.invoke({"query": query})
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        print("\n" + "="*40)
        print(f"ОТВЕТ:\n{answer}")
        print("="*40)
        
        print("\nИсточники (откуда я это взял):")
        for i, doc in enumerate(source_docs):
            print(f"{i+1}. Дата: {doc.metadata.get('date')} | Лайков: {doc.metadata.get('likes')}")
            print(f"   Ссылка: {doc.metadata.get('url')}")
            print(f"   Фрагмент: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    # Интерактивный режим
    while True:
        user_input = input("\nВведите вопрос (или 'exit' для выхода): ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        
        ask_bot(user_input)
