# -*- coding: utf-8 -*-
"""
🔍 WAF EXPLAINER - PHIÊN BẢN ĐÃ FIX
=====================================
CÁC CẢI TIẾN:
1. ✅ FIX: LIME char-level khớp với model tokenization
2. ✅ Thêm explanation cho cả sequence (không chỉ từng ký tự)
3. ✅ Highlight nguy hiểm patterns (SQL keywords, XSS tags)
4. ✅ Better visualization với màu sắc
5. ✅ Export HTML explanation
"""

import torch
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
from model import WAF_Attention_Model
import re
from collections import defaultdict

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
MODEL_PATH = "./data/waf_model.pth"
TOKENIZER_PATH = "./data/tokenizer.pkl"
MAX_LEN = 500
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dangerous patterns để highlight
DANGEROUS_PATTERNS = {
    'SQL': ['select', 'union', 'drop', 'insert', 'delete', 'update', 'exec', 'execute', 
            'waitfor', 'sleep', 'benchmark', 'information_schema', 'xp_cmdshell',
            'or 1=1', 'or 0=0', "' or '", '" or "'],
    'XSS': ['<script', '</script', 'javascript:', 'onerror=', 'onload=', 'onclick=',
            'alert(', 'eval(', 'document.cookie', '<iframe', '<svg', '<img'],
    'CMD': ['etc/passwd', 'etc/shadow', 'win.ini', 'whoami', 'cat ', 'ls ', 
            'rm -rf', 'ping ', 'curl ', 'wget ', '$(', '`', '&&', '||', ';', '|'],
}

# ==============================================================================
# LOAD HỆ THỐNG
# ==============================================================================
def load_system():
    """Load model và tokenizer"""
    print("⏳ Loading model và tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    
    model = WAF_Attention_Model(
        vocab_size=vocab_size, 
        embedding_dim=EMBEDDING_DIM, 
        num_classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded! Vocab size: {vocab_size}")
    return tokenizer, model

tokenizer, model = load_system()

# ==============================================================================
# PREDICTION WRAPPER CHO LIME
# ==============================================================================
def predict_proba(texts):
    """
    Wrapper function cho LIME
    
    LƯU Ý QUAN TRỌNG:
    - LIME sẽ gửi vào các chuỗi đã bị perturb (xóa bớt ký tự)
    - Model vẫn xử lý char-level tokenization như bình thường
    - Điều này đảm bảo LIME explanation chính xác
    """
    # Tokenize từng ký tự (char-level)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    tensor = torch.LongTensor(padded).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = outputs.cpu().numpy()
    
    # Chuyển sang format [P(normal), P(attack)] cho LIME
    results = []
    for p in probs:
        p_attack = p[0]
        p_normal = 1 - p_attack
        results.append([p_normal, p_attack])
    
    return np.array(results)

# ==============================================================================
# CHARACTER-LEVEL LIME EXPLAINER
# ==============================================================================
def create_char_level_explainer():
    """
    Tạo LIME explainer cho character-level
    
    ⚠️ QUAN TRỌNG: char_level=True
    Điều này đảm bảo LIME perturb từng ký tự, khớp với cách model tokenize
    """
    explainer = LimeTextExplainer(
        class_names=["Bình thường", "Tấn công"],
        char_level=True,  # ✅ FIX: Phải là True để khớp với model!
        split_expression=lambda x: list(x),  # Split thành từng ký tự
        bow=False  # Không dùng Bag-of-Words vì character-level
    )
    return explainer

# ==============================================================================
# PATTERN DETECTION
# ==============================================================================
def detect_dangerous_patterns(payload):
    """Phát hiện các pattern nguy hiểm trong payload"""
    payload_lower = payload.lower()
    detected = defaultdict(list)
    
    for category, patterns in DANGEROUS_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in payload_lower:
                detected[category].append(pattern)
    
    return detected

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def colorize_text(text, weights_dict):
    """
    Tô màu text dựa trên importance weights
    
    Màu đỏ: Nguy hiểm (weight > 0)
    Màu xanh: An toàn (weight < 0)
    Màu trắng: Trung tính (weight ≈ 0)
    """
    colored_parts = []
    
    for char in text:
        weight = weights_dict.get(char, 0)
        
        if weight > 0.01:  # Nguy hiểm
            intensity = min(int(abs(weight) * 255), 255)
            colored_parts.append(f"\033[91m{char}\033[0m")  # Red
        elif weight < -0.01:  # An toàn
            intensity = min(int(abs(weight) * 255), 255)
            colored_parts.append(f"\033[92m{char}\033[0m")  # Green
        else:  # Trung tính
            colored_parts.append(char)
    
    return ''.join(colored_parts)

def print_explanation_summary(exp, payload, prediction_proba):
    """In tóm tắt explanation đẹp"""
    print("\n" + "="*70)
    print("🔍 PHÂN TÍCH PAYLOAD")
    print("="*70)
    
    # 1. Payload gốc
    print(f"\n📝 Payload:")
    print(f"   {payload}")
    
    # 2. Dự đoán
    p_normal, p_attack = prediction_proba
    print(f"\n📊 Dự đoán của Model:")
    print(f"   🟢 Bình thường: {p_normal:.2%}")
    print(f"   🔴 Tấn công:    {p_attack:.2%}")
    
    verdict = "⚠️  NGUY HIỂM" if p_attack > 0.5 else "✅ AN TOÀN"
    confidence = max(p_normal, p_attack)
    print(f"   {verdict} (Confidence: {confidence:.2%})")
    
    # 3. Phát hiện patterns
    detected = detect_dangerous_patterns(payload)
    if detected:
        print(f"\n🚨 Phát hiện Pattern Nguy hiểm:")
        for category, patterns in detected.items():
            print(f"   [{category}] {', '.join(patterns)}")
    
    # 4. Top important characters
    print(f"\n💡 Top 15 Ký tự Quan trọng nhất:")
    print("   " + "-"*60)
    
    char_weights = {}
    for char, weight in exp.as_list():
        if char in char_weights:
            char_weights[char] += weight
        else:
            char_weights[char] = weight
    
    # Sort by absolute weight
    sorted_chars = sorted(char_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for i, (char, weight) in enumerate(sorted_chars[:15], 1):
        status = "🔴 Nguy hiểm" if weight > 0 else "🟢 An toàn"
        char_display = repr(char) if char in [' ', '\t', '\n'] else f"'{char}'"
        print(f"   {i:2d}. {char_display:6s} | Weight: {weight:+.4f} ({status})")
    
    # 5. Colored visualization
    print(f"\n🎨 Visualization (Đỏ=Nguy hiểm, Xanh=An toàn):")
    colored = colorize_text(payload, char_weights)
    print(f"   {colored}")
    
    print("="*70)

# ==============================================================================
# NGRAM ANALYSIS (Phân tích theo cụm ký tự)
# ==============================================================================
def analyze_ngrams(payload, exp, n=3):
    """
    Phân tích importance theo n-grams (cụm ký tự)
    
    Ví dụ: "SELECT" có thể được phân tích thành:
    - 3-grams: "SEL", "ELE", "LEC", "ECT"
    - Tổng hợp để hiểu cả cụm "SELECT" nguy hiểm
    """
    char_weights = {}
    for char, weight in exp.as_list():
        char_weights[char] = weight
    
    # Tạo n-grams
    ngrams_weights = {}
    for i in range(len(payload) - n + 1):
        ngram = payload[i:i+n]
        # Tính tổng weight của các ký tự trong ngram
        weight = sum(char_weights.get(c, 0) for c in ngram)
        ngrams_weights[ngram] = weight
    
    # Sort by absolute weight
    sorted_ngrams = sorted(ngrams_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n🔎 Top 10 Cụm {n}-ký tự Nguy hiểm nhất:")
    print("   " + "-"*60)
    for i, (ngram, weight) in enumerate(sorted_ngrams[:10], 1):
        status = "🔴 Attack" if weight > 0 else "🟢 Normal"
        print(f"   {i:2d}. '{ngram}' | Weight: {weight:+.4f} ({status})")

# ==============================================================================
# HTML EXPORT
# ==============================================================================
def export_html_explanation(exp, payload, prediction_proba, filename="explanation.html"):
    """Export explanation ra file HTML để xem trong browser"""
    p_normal, p_attack = prediction_proba
    
    # Tạo HTML với highlighting
    char_weights = {}
    for char, weight in exp.as_list():
        if char in char_weights:
            char_weights[char] += weight
        else:
            char_weights[char] = weight
    
    html_parts = []
    for char in payload:
        weight = char_weights.get(char, 0)
        
        if weight > 0.01:  # Nguy hiểm
            intensity = min(int(abs(weight) * 200) + 55, 255)
            color = f"rgb({intensity}, 0, 0)"
            html_parts.append(f'<span style="background-color: {color}; color: white; padding: 2px;">{char}</span>')
        elif weight < -0.01:  # An toàn
            intensity = min(int(abs(weight) * 200) + 55, 255)
            color = f"rgb(0, {intensity}, 0)"
            html_parts.append(f'<span style="background-color: {color}; color: white; padding: 2px;">{char}</span>')
        else:
            html_parts.append(char)
    
    highlighted_payload = ''.join(html_parts)
    
    # Tạo HTML document
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>WAF Explanation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; max-width: 900px; margin: auto; }}
            h1 {{ color: #333; }}
            .payload {{ background: #f0f0f0; padding: 15px; border-radius: 5px; font-family: monospace; word-wrap: break-word; }}
            .prediction {{ margin: 20px 0; }}
            .bar {{ height: 30px; background: #4CAF50; border-radius: 5px; text-align: center; line-height: 30px; color: white; }}
            .bar.attack {{ background: #f44336; }}
            .legend {{ margin: 20px 0; }}
            .legend span {{ display: inline-block; padding: 5px 10px; margin: 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 WAF Explanation Report</h1>
            
            <h2>📝 Payload</h2>
            <div class="payload">{highlighted_payload}</div>
            
            <h2>📊 Prediction</h2>
            <div class="prediction">
                <p>Normal: {p_normal:.2%}</p>
                <div class="bar" style="width: {p_normal*100}%">{p_normal:.2%}</div>
                
                <p style="margin-top: 10px;">Attack: {p_attack:.2%}</p>
                <div class="bar attack" style="width: {p_attack*100}%">{p_attack:.2%}</div>
            </div>
            
            <div class="legend">
                <h3>Legend:</h3>
                <span style="background: #f44336; color: white;">Đỏ = Nguy hiểm</span>
                <span style="background: #4CAF50; color: white;">Xanh = An toàn</span>
                <span style="background: #f0f0f0;">Trắng = Trung tính</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n💾 Đã lưu explanation ra file: {filename}")

# ==============================================================================
# MAIN EXPLANATION FUNCTION
# ==============================================================================
def explain_payload(payload, num_samples=1000, export_html=False):
    """
    Giải thích tại sao payload được dự đoán là Attack/Normal
    
    Args:
        payload: Chuỗi cần phân tích
        num_samples: Số lượng perturbed samples cho LIME (càng nhiều càng chính xác)
        export_html: Có xuất file HTML không
    """
    # 1. Tạo explainer
    explainer = create_char_level_explainer()
    
    # 2. Explain
    print(f"\n⏳ Đang phân tích payload (num_samples={num_samples})...")
    exp = explainer.explain_instance(
        payload, 
        predict_proba, 
        num_features=len(set(payload)),  # Explain tất cả unique characters
        num_samples=num_samples
    )
    
    # 3. Lấy prediction
    probs = predict_proba([payload])[0]
    
    # 4. In kết quả
    print_explanation_summary(exp, payload, probs)
    
    # 5. N-gram analysis
    analyze_ngrams(payload, exp, n=3)
    analyze_ngrams(payload, exp, n=5)
    
    # 6. Export HTML (optional)
    if export_html:
        export_html_explanation(exp, payload, probs)
    
    return exp, probs

# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🔍 WAF PAYLOAD EXPLAINER - Character-Level Analysis")
    print("="*70)
    print("\nNhập 'exit' để thoát")
    print("Nhập 'html' sau payload để export HTML")
    print("Ví dụ: admin' OR 1=1 -- html")
    print("-"*70)
    
    # Test với một số payload mẫu
    test_payloads = [
        "admin' OR 1=1 --",
        "<script>alert(1)</script>",
        "http://localhost:8000/api/users?id=123",
        "'; DROP TABLE users--",
        "normal_user_search_query",
    ]
    
    print("\n🎯 DEMO: Phân tích một số payload mẫu")
    print("Bạn có muốn xem demo không? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
        for payload in test_payloads:
            print(f"\n{'='*70}")
            print(f"Analyzing: {payload}")
            explain_payload(payload, num_samples=500)
            input("\nNhấn Enter để tiếp tục...")
    
    # Interactive loop
    print("\n" + "="*70)
    print("🔄 INTERACTIVE MODE")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        payload_input = input("\n📝 Nhập payload (hoặc 'exit' để thoát): ").strip()
        
        if payload_input.lower() == 'exit':
            print("👋 Tạm biệt!")
            break
        
        if not payload_input:
            print("⚠️  Vui lòng nhập payload!")
            continue
        
        # Check if user wants HTML export
        export_html = False
        if payload_input.endswith(' html'):
            export_html = True
            payload_input = payload_input[:-5].strip()
        
        try:
            explain_payload(payload_input, num_samples=1000, export_html=export_html)
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()