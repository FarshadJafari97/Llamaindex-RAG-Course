from llama_index.core import VectorStoreIndex, Document , Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import NodeRelationship
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

Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
Settings.llm = GoogleGenAI(model="gemini-3.5-flash-lite")

document = Document(
    text = texts[0],
    metadata = {"source": "LLM", "author": "John Doe"}
)

splitter = SentenceSplitter(chunk_size=86, chunk_overlap=32)

print("Doc ID:", document.doc_id)
print("Doc metadata:", document.metadata)
print("Doc text length:", len(document.text))

nodes = splitter.get_nodes_from_documents([document])
index = VectorStoreIndex(nodes)

# We can also do this
#sentence_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)
#index = VectorStoreIndex.from_documents([document], transformations=[sentence_splitter])

# Nodes info
for node_id, node in index.docstore.docs.items():
    print(f"\nNode ID: {node_id}")
    print(f"  text[:60]: {node.text[:60]}...")
    print(f"  metadata: {node.metadata}")
    print(f"  relationships: {list(node.relationships.keys())}")
    print(f"  embedding dim: {len(node.embedding) if node.embedding else 'None'}")
    print(f"  start_char_idx: {node.start_char_idx}")
    print(f"  end_char_idx: {node.end_char_idx}")



document = Document(
    text = texts[0],
    metadata = {"source": "LLM", "author": "John Doe"}
)

def extend_context(index, node_id, window_size=1):
    current = index.docstore.get_node(node_id)
    before, after = [], []
    
    # Backward
    node = current
    for _ in range(window_size):
        prev_rel = node.relationships.get(NodeRelationship.PREVIOUS)
        if not prev_rel:
            break
        node = index.docstore.get_node(prev_rel.node_id)
        before.insert(0, node.text)
    
    # Forward
    node = current
    for _ in range(window_size):
        next_rel = node.relationships.get(NodeRelationship.NEXT)
        if not next_rel:
            break
        node = index.docstore.get_node(next_rel.node_id)
        after.append(node.text)
    
    return before + [current.text] + after


context = extend_context(index, nodes[0].id_, window_size = 2)

print(context)