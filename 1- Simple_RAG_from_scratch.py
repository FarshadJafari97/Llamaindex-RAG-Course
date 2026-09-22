import os
import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

texts = [
    "فتوسنتز فرآیندی است که در آن گیاهان، جلبک‌ها و برخی باکتری‌ها با استفاده از انرژی نور خورشید، دی‌اکسید کربن و آب را به گلوکز و اکسیژن تبدیل می‌کنند. این فرآیند در کلروپلاست‌ها و با کمک رنگدانه کلروفیل انجام می‌شود. فتوسنتز نقش کلیدی در تولید اکسیژن جو و تأمین انرژی اولیه زنجیره‌های غذایی دارد. بدون فتوسنتز، حیات بر روی زمین به شکلی که می‌شناسیم امکان‌پذیر نبود.",
    "هوش مصنوعی شاخه‌ای از علوم رایانه است که به ساخت سامانه‌هایی می‌پردازد که می‌توانند وظایفی مانند یادگیری، استدلال، ادراک و تصمیم‌گیری را انجام دهند. یادگیری ماشین زیرمجموعه‌ای از هوش مصنوعی است که در آن الگوریتم‌ها با استفاده از داده‌ها الگوها را استخراج می‌کنند. کاربردهای آن شامل تشخیص تصویر، ترجمه ماشینی، پیشنهاددهنده‌ها و خودروهای خودران است. با وجود پیشرفت‌ها، چالش‌هایی مانند سوگیری داده، حریم خصوصی و شفافیت همچنان مطرح است.",
    "دیابت نوع ۲ یک بیماری مزمن است که در آن بدن به دلیل مقاومت به انسولین یا کاهش ترشح انسولین، قند خون را به‌درستی تنظیم نمی‌کند. عوامل خطر شامل چاقی، کم‌تحرکی، رژیم غذایی نامناسب، سابقه خانوادگی و سن بالاست. علائم اولیه ممکن است خفیف باشد و شامل تشنگی زیاد، تکرر ادرار، خستگی و تاری دید شود. مدیریت بیماری بر تغذیه سالم، فعالیت بدنی، کنترل وزن و در صورت نیاز دارو تمرکز دارد.",
    "تغییر اقلیم به تغییرات بلندمدت در دما، بارش، باد و سایر شاخص‌های آب‌وهوایی گفته می‌شود. افزایش گازهای گلخانه‌ای مانند دی‌اکسید کربن و متان، عمدتاً از سوختن سوخت‌های فسیلی، باعث گرمایش جهانی شده است. پیامدها شامل ذوب یخچال‌ها، بالا آمدن سطح دریا، خشکسالی، سیل و تغییر زیستگاه‌هاست. کاهش انتشار گازهای گلخانه‌ای و سازگاری با اثرات اقلیمی از راهبردهای اصلی مقابله با این بحران است.",
    "امپراتوری هخامنشی در حدود سال ۵۵۰ پیش از میلاد توسط کوروش بزرگ بنیان گذاشته شد و به یکی از وسیع‌ترین امپراتوری‌های تاریخ تبدیل شد. این امپراتوری از دریای مدیترانه تا دره سند گسترده بود و با سیستم راه‌ها، چاپارخانه و ساتراپی‌ها اداره می‌شد. منشور کوروش به‌عنوان یکی از نخستین اسناد مربوط به حقوق و آزادی‌های اقوام تحت فرمان شناخته می‌شود. تخت جمشید، پایتخت تشریفاتی آن، نمونه‌ای برجسته از هنر و معماری این دوره است.",
    "عکاسی دیجیتال فرآیند ثبت تصویر با استفاده از حسگر الکترونیکی به‌جای فیلم است. نور از لنز عبور کرده و روی حسگر می‌افتد؛ سپس داده‌ها به سیگنال دیجیتال تبدیل و در حافظه ذخیره می‌شوند. عوامل مؤثر بر کیفیت تصویر شامل اندازه حسگر، دیافراگم، سرعت شاتر و حساسیت ISO است. ویرایش دیجیتال نیز امکان اصلاح نور، رنگ، کنتراست و ترکیب‌بندی را فراهم می‌کند.",
    "خواب یک نیاز زیستی ضروری است که نقش مهمی در تثبیت حافظه، تنظیم خلق‌وخو، تقویت سیستم ایمنی و بازیابی بدن دارد. در طول خواب، مغز اطلاعات روز را پردازش و دسته‌بندی می‌کند و ارتباطات عصبی مرتبط با یادگیری تقویت می‌شود. کم‌خوابی مزمن با مشکلاتی مانند کاهش تمرکز، تحریک‌پذیری، ضعف ایمنی و افزایش خطر بیماری‌های قلبی مرتبط است. رعایت بهداشت خواب، شامل ساعت منظم، محیط تاریک و پرهیز از صفحه‌نمایش پیش از خواب، کیفیت خواب را بهبود می‌دهد.",
    "اقتصاد رفتاری حوزه‌ای میان‌رشته‌ای است که نقش عوامل روان‌شناختی، اجتماعی و شناختی را در تصمیم‌های اقتصادی بررسی می‌کند. برخلاف مدل انسان کاملاً منطقی، این حوزه نشان می‌دهد افراد تحت تأثیر سوگیری‌هایی مانند لنگر انداختن، زیان‌گریزی و اثر چارچوب‌بندی قرار می‌گیرند. این یافته‌ها در طراحی سیاست‌های عمومی، بازاریابی، سرمایه‌گذاری و معماری انتخاب کاربرد دارند. مثال معروف آن، افزایش نرخ پس‌انداز بازنشستگی با ثبت خودکار افراد در طرح‌هاست.",
    "معماری پایدار رویکردی در طراحی ساختمان است که هدف آن کاهش مصرف انرژی، آب و منابع طبیعی و کمینه‌سازی اثرات زیست‌محیطی در طول چرخه عمر بناست. این رویکرد از عوامل اقلیمی، جهت‌گیری ساختمان، عایق‌بندی، تهویه طبیعی، نور روز و انرژی‌های تجدیدپذیر بهره می‌گیرد. استفاده از مصالح بازیافتی یا محلی و کاهش تولید زباله نیز بخشی از اصول آن است. ساختمان‌های پایدار می‌توانند هزینه‌های بهره‌برداری را کاهش دهند و کیفیت زندگی ساکنان را بهبود بخشند.",
    "ورزش هوازی فعالیتی است که ضربان قلب و تنفس را برای مدت نسبتاً طولانی افزایش می‌دهد و عضلات بزرگ بدن را درگیر می‌کند. نمونه‌هایی از آن شامل دویدن، شنا، دوچرخه‌سواری و پیاده‌روی سریع است. این نوع ورزش به بهبود سلامت قلب و عروق، افزایش ظرفیت ریه، کنترل وزن، کاهش فشار خون و تقویت خلق‌وخو کمک می‌کند. توصیه عمومی آن است که بزرگسالان در هفته حداقل ۱۵۰ دقیقه فعالیت هوازی با شدت متوسط یا ۷۵ دقیقه با شدت شدید داشته باشند."
]

class GeminiEmbedder:
    def __init__(self, model="gemini-embedding-2", api_key=None):
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model

    def embed_documents(self, texts):
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values


class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = []

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query_vector, top_k=3):
        if not self.vectors:
            return []
        matrix = np.array(self.vectors)
        q = np.array(query_vector).reshape(1, -1)
        scores = cosine_similarity(q, matrix)[0]
        ranked = sorted(
            zip(self.chunks, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


class RAG:
    def __init__(self, embedder, llm_fn):
        self.embedder = embedder
        self.llm_fn = llm_fn
        self.store = SimpleVectorStore()

    def index(self, documents):
        vectors = self.embedder.embed_documents(documents)
        self.store.add(documents, vectors)

    def _build_prompt(self, question, retrieved):
        context = "\n\n".join(
            f"[منبع {i+1} | امتیاز {score:.3f}]\n{chunk}"
            for i, (chunk, score) in enumerate(retrieved)
        )
        return f"""فقط بر اساس متن‌های زیر پاسخ بده.
اگر پاسخ در متن‌ها نبود، صریحاً بگو: "بر اساس اسناد ارائه‌شده نمی‌دانم."

متن‌ها:
{context}

پرسش: {question}

پاسخ:"""

    def query(self, question, top_k=3, score_threshold=None):
        q_vec = self.embedder.embed_query(question)
        retrieved = self.store.search(q_vec, top_k=top_k)

        if score_threshold is not None and (
            not retrieved or retrieved[0][1] < score_threshold
        ):
            return {
                "answer": "بر اساس اسناد ارائه‌شده نمی‌دانم.",
                "sources": retrieved,
                "reason": f"best_score={retrieved[0][1]:.3f} < threshold={score_threshold}"
                if retrieved else "no results",
            }

        prompt = self._build_prompt(question, retrieved)
        answer = self.llm_fn(prompt)
        return {"answer": answer, "sources": retrieved}


def make_llm_fn(client, model="gemini-3.5-flash-lite"):
    def llm_fn(prompt):
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.candidates[0].content.parts[0].text
    return llm_fn


# --- usage ---
if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    embedder = GeminiEmbedder(api_key=os.environ["GEMINI_API_KEY"])
    rag = RAG(embedder=embedder, llm_fn=make_llm_fn(client))

    rag.index(texts)

    for q in [
        "تغییرات اقلیمی چه پیامدهایی دارد؟",
        "پایتخت فرانسه کجاست؟",
        "فتوسنتز در کدام اندامک انجام می‌شود؟",
    ]:
        result = rag.query(q, top_k=3, score_threshold=0.5)
        print("=" * 60)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        for chunk, score in result["sources"]:
            print(f"  [{score:.3f}] {chunk[:60]}...")