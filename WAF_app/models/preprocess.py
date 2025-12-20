import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import urllib.parse
import os
import re

# =========================================================
# CẤU HÌNH - TỐI ƯU CHO >96% ACCURACY
# =========================================================
CSIC_PATH = './data/CSIC_2010/csic_database.csv'
HTTP_PATH = './data/HTTPParams_2015/payload_full.csv'
CUSTOM_PATH = './data/custom_data.csv'

MAX_VOCAB = 10000
MAX_LEN = 512  # Tăng lên 512 để bắt trọn payload dài hơn
OUTPUT_DATA = "./data/processed_data.pkl"
OUTPUT_TOKENIZER = "./data/tokenizer.pkl"

# =========================================================
# CÁC HÀM TIỀN XỬ LÝ NÂNG CAO
# =========================================================

def advanced_url_decode(text, max_iterations=3):
    """
    Giải mã URL nhiều lớp (xử lý double/triple encoding)
    """
    if not isinstance(text, str):
        return str(text)

    result = text
    for _ in range(max_iterations):
        try:
            decoded = urllib.parse.unquote(result)
            if decoded == result:
                break
            result = decoded
        except:
            break
    return result

def normalize_whitespace(text):
    """Chuẩn hóa whitespace"""
    # Thay nhiều space thành 1 space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_payload(text):
    """
    Làm sạch payload:
    - Decode URL
    - Normalize whitespace
    - Giữ lại các ký tự quan trọng cho attack detection
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. URL Decode (multi-layer)
    text = advanced_url_decode(text)

    # 2. Normalize whitespace
    text = normalize_whitespace(text)

    # 3. Lowercase để model dễ học hơn
    text = text.lower()

    return text

def remove_duplicates(df):
    """Loại bỏ các dòng trùng lặp"""
    original_len = len(df)
    df = df.drop_duplicates(subset=['text'])
    removed = original_len - len(df)
    if removed > 0:
        print(f"   Đã loại bỏ {removed:,} dòng trùng lặp")
    return df

def print_class_distribution(df):
    """
    In thong ke phan bo class (KHONG cat giam data)
    Focal Loss trong model se xu ly class imbalance
    """
    attack_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    total = len(df)

    print(f"\n📊 Phan bo class:")
    print(f"   Normal: {normal_count:,} ({normal_count/total:.1%})")
    print(f"   Attack: {attack_count:,} ({attack_count/total:.1%})")
    print(f"   Focal Loss se xu ly class imbalance trong qua trinh training.")

    return df

# =========================================================
# CÁC HÀM LOAD DỮ LIỆU
# =========================================================

def load_csic_by_index(filepath):
    """Load CSIC 2010 dựa trên vị trí cột"""
    print(f"Loading CSIC 2010: {filepath}...")
    if not os.path.exists(filepath):
        print(f"❌ Lỗi: Không tìm thấy file {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, header=None, on_bad_lines='skip', low_memory=False)

        # Bỏ dòng header rác nếu có
        first_val = str(df.iloc[0, 1])
        if "Method" in first_val or "User-Agent" in first_val:
            df = df.iloc[1:]

        df_clean = pd.DataFrame()
        df_clean['text'] = df.iloc[:, -1].fillna('').astype(str)

        def clean_label(val):
            return 0 if 'Normal' in str(val) else 1

        df_clean['label'] = df.iloc[:, 0].apply(clean_label)

        print(f" -> CSIC Loaded: {len(df_clean):,} dòng.")
        return df_clean
    except Exception as e:
        print(f"❌ Lỗi đọc CSIC: {e}")
        return pd.DataFrame()

def load_httpparams(filepath):
    """Load HTTPParams 2015"""
    print(f"Loading HTTPParams: {filepath}...")
    if not os.path.exists(filepath):
        print(f"❌ Lỗi: Không tìm thấy file {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df = df[['payload', 'label']].rename(columns={'payload': 'text'})

        def clean_label(val):
            val = str(val).strip().lower()
            return 0 if val == 'norm' else 1

        df['label'] = df['label'].apply(clean_label)

        print(f" -> HTTPParams Loaded: {len(df):,} dòng.")
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc HTTPParams: {e}")
        return pd.DataFrame()

def load_custom(filepath):
    """Load dữ liệu tự sinh"""
    print(f"Loading Custom Data: {filepath}...")
    if not os.path.exists(filepath):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {filepath}. Hãy chạy gen_data.py trước!")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        print(f" -> Custom Data Loaded: {len(df):,} dòng.")
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc Custom Data: {e}")
        return pd.DataFrame()

# =========================================================
# MAIN PROCESS
# =========================================================

def main():
    print("="*60)
    print("🔧 DATA PREPROCESSING - TỐI ƯU CHO >96% ACCURACY")
    print("="*60)

    # 1. Gộp dữ liệu
    df1 = load_csic_by_index(CSIC_PATH)
    df2 = load_httpparams(HTTP_PATH)
    df3 = load_custom(CUSTOM_PATH)

    full_df = pd.concat([df1, df2, df3], ignore_index=True)

    if len(full_df) == 0:
        print("❌ Không có dữ liệu nào để xử lý!")
        return

    print(f"\n📊 Tổng dữ liệu trước xử lý: {len(full_df):,} mẫu")

    # 2. Làm sạch text
    print("\n🧹 Đang làm sạch dữ liệu...")
    full_df['text'] = full_df['text'].apply(clean_payload)

    # 3. Loại bỏ các dòng rỗng hoặc quá ngắn
    min_length = 3
    full_df = full_df[full_df['text'].str.len() >= min_length]
    print(f"   Sau khi loại payload quá ngắn: {len(full_df):,} mẫu")

    # 4. Loại bỏ duplicates
    full_df = remove_duplicates(full_df)

    # 5. In thong ke phan bo (KHONG cat giam data)
    full_df = print_class_distribution(full_df)

    # 6. Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 7. Thống kê cuối cùng
    counts = full_df['label'].value_counts()
    print(f"\n📊 THỐNG KÊ CUỐI CÙNG:")
    print(f"   Tổng mẫu: {len(full_df):,}")
    print(f"   Normal (0): {counts.get(0, 0):,}")
    print(f"   Attack (1): {counts.get(1, 0):,}")
    print(f"   Tỉ lệ Attack: {counts.get(1, 0) / len(full_df) * 100:.2f}%")

    # 8. Tokenization (Char-level)
    print("\n🔤 Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, char_level=True, lower=False, oov_token='<UNK>')
    tokenizer.fit_on_texts(full_df['text'])
    sequences = tokenizer.texts_to_sequences(full_df['text'])

    print(f"   Vocabulary size: {len(tokenizer.word_index):,} characters")

    # 9. Padding
    print(f"📏 Padding to {MAX_LEN}...")
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y = np.array(full_df['label']).astype(np.float32)

    # 10. Stratified Split (QUAN TRỌNG!)
    print("\n✂️ Stratified Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # QUAN TRỌNG: Đảm bảo phân bố class giống nhau trong train/test
    )

    # Verify stratification
    train_attack_ratio = y_train.mean()
    test_attack_ratio = y_test.mean()
    print(f"   Train: {len(X_train):,} mẫu (Attack ratio: {train_attack_ratio:.2%})")
    print(f"   Test:  {len(X_test):,} mẫu (Attack ratio: {test_attack_ratio:.2%})")

    # 11. Lưu file
    data_package = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "vocab_size": len(tokenizer.word_index) + 1,
        "max_len": MAX_LEN
    }

    os.makedirs(os.path.dirname(OUTPUT_DATA), exist_ok=True)

    with open(OUTPUT_DATA, 'wb') as f:
        pickle.dump(data_package, f)
    with open(OUTPUT_TOKENIZER, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"\n✅ XONG!")
    print(f"   Đã lưu dữ liệu: {OUTPUT_DATA}")
    print(f"   Đã lưu tokenizer: {OUTPUT_TOKENIZER}")
    print("\n👉 Chạy 'python train.py' để huấn luyện mô hình!")

if __name__ == "__main__":
    main()
