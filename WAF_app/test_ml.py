"""Test ML Predictor with PyTorch model"""
from ml_predictor import get_ml_predictor

# Test cases
test_cases = [
    ("SQL Injection", "admin' OR '1'='1"),
    ("SQL Injection", "SELECT * FROM users WHERE id = 1 OR 1=1"),
    ("SQL Injection", "1' UNION SELECT password FROM users--"),
    ("XSS", "<script>alert('XSS')</script>"),
    ("XSS", "<img src=x onerror=alert(1)>"),
    ("Path Traversal", "../../../etc/passwd"),
    ("Command Injection", "; cat /etc/passwd"),
    ("Command Injection", "$(whoami)"),
    ("Normal", "GET /index.html"),
    ("Normal", "/products?id=123"),
    ("Normal", "login=user1&password=pass123"),  # Normal form data
]

predictor = get_ml_predictor()
print('=' * 70)
print(f'Model loaded: {predictor.is_loaded}')
print('=' * 70)

for attack_type, payload in test_cases:
    # predict() may return 2 or 3 values (with XAI patterns)
    result = predictor.predict(payload)
    if len(result) == 3:
        pred, conf, patterns = result
    else:
        pred, conf = result
        patterns = {}
    
    status = "✓" if (pred == 'attack' and attack_type != 'Normal') or (pred == 'normal' and attack_type == 'Normal') else "✗"
    
    print(f"\n{status} [{attack_type}]")
    print(f"  Payload: {payload}")
    print(f"  Prediction: {pred.upper()} (confidence: {conf:.4f})")
    if patterns:
        print(f"  XAI Patterns: {patterns}")
