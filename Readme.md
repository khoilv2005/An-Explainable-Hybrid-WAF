# 🛡️ WAF - Web Application Firewall with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-orange.svg)](https://mysql.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống **Web Application Firewall (WAF)** kết hợp rule-based detection và **Deep Learning** để phát hiện các cuộc tấn công web. Sử dụng mô hình PyTorch với kiến trúc Attention và **LIME XAI** để giải thích các quyết định.

---

## 📑 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Sử dụng](#-sử-dụng)
- [API Endpoints](#-api-endpoints)
- [Admin Panel](#-admin-panel)
- [Machine Learning](#-machine-learning)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)

---

## ✨ Tính năng

### Rule-based Detection
- ✅ **SQL Injection** - Phát hiện các mẫu SQL injection phổ biến
- ✅ **XSS (Cross-Site Scripting)** - Ngăn chặn các script độc hại
- ✅ **Path Traversal** - Chặn truy cập thư mục trái phép
- ✅ **Command Injection** - Phát hiện lệnh shell injection
- ✅ **IP Blacklist** - Tự động chặn IP vi phạm nhiều lần

### Deep Learning Detection
- 🤖 **Deep Learning Model** - Mô hình PyTorch với Attention mechanism
- 🚀 **ONNX Runtime** - Inference nhanh với ONNX optimization  
- 🔍 **LIME XAI** - Giải thích lý do phát hiện tấn công
- 📊 **Character-level Tokenization** - Phát hiện payload ẩn

### Admin Panel
- 📈 **Dashboard** - Giám sát real-time các hoạt động
- 📝 **Rule Management** - Thêm/Sửa/Xóa rules
- 🚫 **IP Blacklist** - Quản lý danh sách IP bị chặn
- 📄 **Activity Logs** - Xem lịch sử các request với pagination

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│     Client      │────▶│   WAF Proxy     │────▶│  Backend App    │
│                 │     │  (Port 8080)    │     │  (Your App)     │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
              │   Rule    │ │   ML    │ │   MySQL   │
              │  Engine   │ │ Engine  │ │  Database │
              └───────────┘ └─────────┘ └───────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │   Admin Panel     │
                                    │   (Port 5000)     │
                                    └───────────────────┘
```

---

## 🚀 Cài đặt

### Yêu cầu
- **Docker** & **Docker Compose**
- **Git**

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd NT140.Q11-D03-WAF
```

### Bước 2: Cấu hình environment

```bash
cp .env.example .env
```

Chỉnh sửa file `.env` theo nhu cầu (xem phần [Cấu hình](#-cấu-hình)).

### Bước 3: Chạy với Docker Compose

```bash
docker-compose up -d
```

### Bước 4: Kiểm tra services

```bash
# Kiểm tra các container đang chạy
docker-compose ps

# Xem logs
docker-compose logs -f waf_app
docker-compose logs -f waf_admin
```

---

## ⚙️ Cấu hình

### File `.env`

| Biến | Mô tả | Giá trị mặc định |
|------|-------|------------------|
| `MYSQL_ROOT_PASSWORD` | Mật khẩu root MySQL | `rootpassword` |
| `MYSQL_DATABASE` | Tên database | `wafdb` |
| `MYSQL_USER` | Username MySQL | `waf` |
| `MYSQL_PASSWORD` | Password MySQL | `wafadmin` |
| `WAF_LISTEN_PORT` | Port WAF lắng nghe | `8080` |
| `WAF_BACKEND_ADDRESS` | Địa chỉ backend app | `http://host.docker.internal:8888` |
| `WAF_BLOCK_THRESHOLD` | Ngưỡng block IP | `100000` |
| `WAF_ML_ENABLED` | Bật/tắt ML detection | `true` |
| `WAF_ML_CONFIDENCE_THRESHOLD` | Ngưỡng tin cậy ML | `0.5` |
| `WAF_ML_LIME_ENABLED` | Bật/tắt LIME XAI | `false` |
| `ADMIN_LISTEN_PORT` | Port Admin Panel | `5000` |
| `ADMIN_SECRET_KEY` | Secret key Flask | `super_secret_key...` |
| `ADMIN_ALLOWED_IPS` | Danh sách IP được phép | `127.0.0.1,::1` |

### Cấu hình ML

```bash
# Bật ML detection
WAF_ML_ENABLED=true

# Ngưỡng confidence (0.0 - 1.0)
# Thấp hơn = nhạy hơn, cao hơn = ít false positive
WAF_ML_CONFIDENCE_THRESHOLD=0.5

# Bật LIME explanations (ảnh hưởng hiệu năng)
WAF_ML_LIME_ENABLED=false
```

---

## 📖 Sử dụng

### Truy cập các services

| Service | URL | Mô tả |
|---------|-----|-------|
| **WAF Proxy** | `http://localhost:8080` | Reverse proxy WAF |
| **Admin Panel** | `http://localhost:5000` | Quản lý WAF |
| **MySQL** | `localhost:3306` | Database |

### Test WAF với curl

```bash
# Request hợp lệ
curl http://localhost:8080/

# Test SQL Injection (sẽ bị chặn)
curl "http://localhost:8080/?id=1' OR '1'='1"

# Test XSS (sẽ bị chặn)
curl "http://localhost:8080/?q=<script>alert(1)</script>"

# Test Path Traversal (sẽ bị chặn)
curl "http://localhost:8080/../../../etc/passwd"
```

---

## 🔌 API Endpoints

### WAF Application (Port 8080)

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/{path:path}` | ALL | Reverse proxy đến backend |
| `/health` | GET | Health check endpoint |
| `/reset-db-management` | POST | Reload rules từ database |

### Admin Panel (Port 5000)

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Dashboard chính |
| `/api/logs/latest` | GET | API lấy logs (AJAX) |
| `/rules` | GET | Xem danh sách rules |
| `/rules/add` | GET, POST | Thêm rule mới |
| `/rules/delete/<id>` | POST | Xóa rule |
| `/rules/delete-all` | POST | Xóa tất cả rules |
| `/rules/import` | POST | Import rules từ JSON |
| `/blacklist` | GET | Xem IP blacklist |
| `/blacklist/remove/<ip>` | POST | Xóa IP khỏi blacklist |
| `/reset-all` | POST | Reset toàn bộ dữ liệu |

---

## 🖥️ Admin Panel

### Dashboard
- Xem thống kê tổng quan (tổng requests, blocked, allowed)
- Activity log real-time với auto-refresh
- Biểu đồ phân tích

### Quản lý Rules
- Thêm/sửa/xóa rules
- Import rules từ file JSON
- Phân loại theo category: SQLi, XSS, Path Traversal, etc.

### IP Blacklist
- Xem danh sách IP bị chặn
- Xóa IP khỏi blacklist
- Xem rule trigger gây block

---

## 🤖 Deep Learning

### Tổng quan

Hệ thống sử dụng mô hình **Deep Learning** tùy chỉnh được xây dựng bằng **PyTorch**, kết hợp nhiều kỹ thuật tiên tiến để phát hiện các cuộc tấn công web với độ chính xác cao.

### Kiến trúc Model: WAF_Attention_Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (Character-level)                     │
│                    Max Length: 500 characters                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EMBEDDING LAYER                            │
│              Vocab Size → Embedding Dim (128)                   │
│                    + Dropout (0.1)                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               CNN FEATURE EXTRACTION                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ ResBlock 1  │→ │ ResBlock 2  │→ │ ResBlock 3  │             │
│  │  128 → 128  │  │  128 → 256  │  │  256 → 256  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         + Squeeze-and-Excitation (SE) Attention                 │
│         + MaxPool + Dropout                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-HEAD SELF-ATTENTION                          │
│                    8 Attention Heads                            │
│              + Layer Normalization                              │
│              + Residual Connections                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               BI-DIRECTIONAL LSTM                               │
│                  2 Layers, 256 Hidden                           │
│                  + Attention Pooling                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                CLASSIFICATION HEAD                              │
│    Dense(512→256) → GELU → Dense(256→128) → Dense(128→1)       │
│              + Layer Norm + Dropout                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│           Sigmoid → Probability (0: Normal, 1: Attack)          │
└─────────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

| Component | Mô tả |
|-----------|-------|
| **Multi-Head Self-Attention** | 8 heads để capture các mối quan hệ giữa các ký tự trong payload |
| **Residual Blocks** | Skip connections giúp training deep network hiệu quả |
| **SE Block (Squeeze-Excitation)** | Channel attention để tập trung vào features quan trọng |
| **Bi-LSTM** | Capture context từ cả 2 hướng của sequence |
| **GELU Activation** | Smooth activation function, hiệu quả hơn ReLU |
| **Layer Normalization** | Stabilize training và tăng tốc convergence |

### Training Pipeline

| Kỹ thuật | Chi tiết |
|----------|----------|
| **Loss Function** | Focal Loss (α=0.25, γ=2.0) - xử lý class imbalance |
| **Optimizer** | AdamW với weight decay |
| **Label Smoothing** | 0.1 - giúp model generalize tốt hơn |
| **Tokenization** | Character-level - phát hiện payload obfuscation |
| **Mixed Precision** | FP16 training cho tốc độ cao hơn |

### ONNX Optimization

Model được export sang **ONNX** format để inference nhanh hơn trong production:

```python
# Inference với ONNX Runtime
ML_FORCE_ONNX=true
ML_MODEL_ONNX_PATH=/app/models/waf_model.onnx
```

| Metric | PyTorch | ONNX Runtime |
|--------|---------|--------------|
| **Latency** | ~15ms | ~3ms |
| **Memory** | ~500MB | ~150MB |
| **Throughput** | ~65 req/s | ~300 req/s |

### Explainable AI (XAI) với LIME

Khi `WAF_ML_LIME_ENABLED=true`, hệ thống sử dụng **LIME (Local Interpretable Model-agnostic Explanations)** để:

1. **Giải thích quyết định** - Highlight các token đóng góp vào prediction
2. **Phát hiện patterns** - Tự động detect SQL, XSS, command injection patterns
3. **Debug & Audit** - Log chi tiết lý do block request

```
Example LIME Output:
─────────────────────────────────────
Request: /search?q=1' OR '1'='1
Prediction: ATTACK (confidence: 0.98)

Top contributing tokens:
  [+0.45] OR
  [+0.32] '1'='1
  [+0.21] '
─────────────────────────────────────
```

### Files & Models

| File | Kích thước | Mô tả |
|------|------------|-------|
| `waf_model.onnx` | ~15MB | ONNX model cho production |
| `waf_model.pth` | ~15MB | PyTorch checkpoint |
| `tokenizer_word_index.json` | ~1KB | Character vocabulary |
| `model.py` | - | Model architecture definition |
| `train.py` | - | Training script |
| `preprocess.py` | - | Data preprocessing |

### Cấu hình Deep Learning

```bash
# Bật/tắt Deep Learning detection
WAF_ML_ENABLED=true

# Ngưỡng confidence (0.0 - 1.0)
# Cao hơn = ít false positive, thấp hơn = detect nhiều hơn
WAF_ML_CONFIDENCE_THRESHOLD=0.5

# Bật LIME explanations (tăng latency ~100ms)
WAF_ML_LIME_ENABLED=false

# Force sử dụng ONNX (khuyến nghị cho production)
ML_FORCE_ONNX=true
```

---

## 📁 Cấu trúc dự án

```
NT140.Q11-D03-WAF/
├── 📂 WAF_app/                   # WAF Application
│   ├── main.py                   # FastAPI reverse proxy
│   ├── ml_predictor.py           # ML inference engine
│   ├── decoder.py                # URL/HTML decoder
│   ├── Dockerfile
│   └── 📂 models/                # ML models
│       ├── waf_model.onnx
│       └── tokenizer_word_index.json
│
├── 📂 WAF_admin/                 # Admin Panel
│   ├── main.py                   # Flask application
│   ├── Dockerfile
│   └── 📂 templates/             # HTML templates
│       └── admin_dashboard.html
│
├── 📂 shared/                    # Shared code
│   ├── models.py                 # SQLAlchemy models
│   └── database.py               # Database connection
│
├── 📂 rules/                     # WAF rules
│   └── complete_rules_import.json
│
├── docker-compose.yml            # Docker Compose config
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## 📊 Database Schema

### Tables

| Table | Mô tả |
|-------|-------|
| `rules` | Các rule WAF |
| `ip_blacklist` | Danh sách IP bị chặn |
| `activity_log` | Log hoạt động |

### Rule Structure

```json
{
  "id": 1,
  "enabled": true,
  "description": "SQL Injection - Basic",
  "category": "SQLi",
  "severity": "HIGH",
  "target": "REQUEST_URI",
  "operator": "rx",
  "value": "(?i)(union\\s+select|select.*from)",
  "action": "BLOCK"
}
```

---

## 🔧 Docker Commands

```bash
# Chạy services
docker-compose up -d

# Dừng services
docker-compose down

# Xem logs
docker-compose logs -f

# Rebuild containers
docker-compose up -d --build

# Vào container
docker exec -it waf_app bash
docker exec -it waf_admin bash
docker exec -it waf_mysql mysql -u waf -p
```

---

## 📝 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👥 Contributors

- **NT140.Q11 - Nhóm 6** - University of Information Technology (UIT)

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Flask](https://flask.palletsprojects.com/) - Lightweight Python web framework
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference
- [LIME](https://github.com/marcotcr/lime) - Explainable AI
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL toolkit
