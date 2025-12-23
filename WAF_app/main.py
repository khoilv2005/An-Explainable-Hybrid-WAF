import sys
import os
import re
import json
import ipaddress
import asyncio
from functools import lru_cache
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from shared.database import Rule, IPBlacklist, ActivityLog, logger, init_database
    from shared.models import Base
    import shared.models  # Ensure models are loaded
except ImportError:
    print("FATAL ERROR: Could not import from 'shared/database.py' or 'shared/models.py'.")
    sys.exit(1)

try:
    from decoder import deep_decode_data
except ImportError:
    print("FATAL ERROR: Could not import from 'decoder.py'.")
    sys.exit(1)

try:
    from ml_predictor import get_ml_predictor
    ML_AVAILABLE = True
except ImportError:
    print("WARNING: ML predictor not available. WAF will run with rule-based detection only.")
    ML_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Configuration
BACKEND_ADDRESS = os.getenv("WAF_BACKEND_ADDRESS", "http://127.0.0.1:80")
LISTEN_HOST = os.getenv("WAF_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("WAF_LISTEN_PORT", "8080"))

ML_ENABLED = os.getenv("WAF_ML_ENABLED", "true").lower() == "true"
ML_CONFIDENCE_THRESHOLD = float(os.getenv("WAF_ML_CONFIDENCE_THRESHOLD", "0.7"))
ML_LIME_ENABLED = os.getenv("WAF_ML_LIME_ENABLED", "true").lower() == "true"
BLOCK_THRESHOLD = int(os.getenv("WAF_BLOCK_THRESHOLD", "100000"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# SQLAlchemy async setup
async_engine = create_async_engine(
    DATABASE_URL.replace("mysql+pymysql", "mysql+aiomysql"), 
    echo=False, pool_size=20, max_overflow=0
)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Global caches
WAF_RULES = []
IP_BLACKLIST = frozenset()
_COMPILED_REGEX_CACHE = {}

# ====== Lifecycle ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await load_cache_from_db()
    logger.info(f"WAF Service (FastAPI/Async) is running on http://{LISTEN_HOST}:{LISTEN_PORT}")
    yield
    # Shutdown
    await async_engine.dispose()

app = FastAPI(lifespan=lifespan)

# ====== Database helpers ======
async def init_db():
    """Initialize database schema with retry logic"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Database initialization failed after {max_retries} attempts: {e}")
                raise

@lru_cache(maxsize=512)
def get_compiled_regex(pattern: str):
    """Cache compiled regex patterns"""
    return re.compile(pattern, re.IGNORECASE)

async def load_cache_from_db():
    """Load rules and IP blacklist from DB into cache (async)"""
    global WAF_RULES, IP_BLACKLIST
    
    try:
        async with AsyncSessionLocal() as session:
            # Load rules
            rules_query = select(Rule).where(Rule.enabled == True)
            result = await session.execute(rules_query)
            rules_obj = result.scalars().all()
            new_rules = [r.to_dict() for r in rules_obj]
            
            for rule in new_rules:
                if rule.get('operator') in ('REGEX', 'REGEX_MATCH'):
                    pattern = rule.get('value', '')
                    if pattern:
                        try:
                            get_compiled_regex(pattern)
                        except re.error as e:
                            logger.error(f"Invalid regex in rule {rule['id']}: {e}")
            
            WAF_RULES = new_rules
            logger.info(f"Loaded {len(WAF_RULES)} active rules")
            
            # Load IP blacklist
            ips_query = select(IPBlacklist.ip_address)
            result = await session.execute(ips_query)
            ips_list = result.scalars().all()
            IP_BLACKLIST = frozenset(ips_list)
            logger.info(f"Loaded {len(IP_BLACKLIST)} blacklisted IPs")
            
    except Exception as e:
        logger.error(f"Failed to load cache from database: {e}")

async def log_event_to_db(client_ip, method, path, status, action, rule_id=None):
    """Log event to DB asynchronously"""
    try:
        async with AsyncSessionLocal() as session:
            new_log = ActivityLog(
                client_ip=client_ip, request_method=method, request_path=path,
                status_code=status, action_taken=action, triggered_rule_id=rule_id
            )
            session.add(new_log)
            await session.commit()
    except Exception as e:
        # Handle foreign key constraint error - rule may not exist in DB yet
        if "foreign key constraint" in str(e).lower() or "1452" in str(e):
            logger.warning(f"Rule ID {rule_id} not found in DB, logging without rule reference")
            try:
                async with AsyncSessionLocal() as session:
                    new_log = ActivityLog(
                        client_ip=client_ip, request_method=method, request_path=path,
                        status_code=status, action_taken=action, triggered_rule_id=None
                    )
                    session.add(new_log)
                    await session.commit()
            except Exception as e2:
                logger.error(f"DB Log Failed (retry): {e2}")
        else:
            logger.error(f"DB Log Failed: {e}")

async def add_ip_to_blacklist(ip: str, rule_id: int):
    """Add IP to blacklist asynchronously"""
    global IP_BLACKLIST
    
    try:
        async with AsyncSessionLocal() as session:
            # Check if already exists
            existing_query = select(IPBlacklist).where(IPBlacklist.ip_address == ip)
            result = await session.execute(existing_query)
            existing_ip = result.scalar()
            
            if not existing_ip:
                new_blacklist_entry = IPBlacklist(
                    ip_address=ip,
                    triggered_rule_id=rule_id,
                    notes=f"Auto-blocked after reaching {BLOCK_THRESHOLD} violations."
                )
                session.add(new_blacklist_entry)
                await session.commit()
                logger.info(f"IP {ip} has been added to the blacklist.")
                
                # Update cache
                IP_BLACKLIST = IP_BLACKLIST | {ip}
                    
    except Exception as e:
        logger.error(f"Could not add IP {ip} to blacklist: {e}")

async def check_and_auto_block(ip: str, rule_id: int):
    """Check IP violation history and auto-block if needed"""
    try:
        async with AsyncSessionLocal() as session:
            block_query = select(ActivityLog).where(
                ActivityLog.client_ip == ip,
                ActivityLog.action_taken == 'BLOCKED'
            )
            result = await session.execute(block_query)
            block_count = len(result.scalars().all())
            
            logger.info(f"IP {ip} has {block_count} previous blocks. Threshold is {BLOCK_THRESHOLD}.")
            
            if block_count >= BLOCK_THRESHOLD - 1:
                await add_ip_to_blacklist(ip, rule_id)
    except Exception as e:
        logger.error(f"Could not check auto-block status for IP {ip}: {e}")

# ====== Rule inspection ======
def evaluate_operator(operator: str, value: str, target: str) -> bool:
    """Evaluate operator for rule matching"""
    try:
        if operator == 'CONTAINS':
            return value in target
        if operator in ('REGEX', 'REGEX_MATCH'):
            compiled = get_compiled_regex(value)
            return bool(compiled.search(target))
        if operator == '@eq':
            return str(target) == str(value)
        if operator == '@gt':
            return float(target) > float(value)
        if operator == '@lt':
            return float(target) < float(value)
        if operator == '@ipMatch':
            return _check_ip_match(target, value)
        logger.warning(f"Unknown operator: {operator}")
        return False
    except (ValueError, TypeError, re.error) as e:
        logger.debug(f"Operator evaluation error for {operator}: {e}")
        return False

def _check_ip_match(target: str, value: str) -> bool:
    """Helper for IP/CIDR matching"""
    try:
        target_ip = ipaddress.ip_address(str(target))
        for val_ip in value.split(','):
            val_ip = val_ip.strip()
            if not val_ip:
                continue
            if '/' in val_ip:
                if target_ip in ipaddress.ip_network(val_ip, strict=False):
                    return True
            elif target_ip == ipaddress.ip_address(val_ip):
                return True
        return False
    except (ValueError, ipaddress.AddressValueError):
        return False

def parse_structured_data(data: str) -> list:
    """Parse JSON/XML data"""
    if not data or len(data) < 2:
        return []
    
    parsed_values = []
    data_stripped = data.strip()
    
    if data_stripped.startswith(('{', '[')):
        try:
            json_obj = json.loads(data)
            parsed_values.extend(_extract_values_from_dict(json_obj))
        except (json.JSONDecodeError, ValueError):
            pass
    
    if '<' in data and '>' in data:
        xml_tags = re.findall(r'<([^!/?][^>]*)>([^<]*)</\1>', data)
        for _, content in xml_tags:
            content_stripped = content.strip()
            if content_stripped:
                parsed_values.append(content_stripped)
    
    return parsed_values

def _extract_values_from_dict(obj, _depth=0) -> list:
    """Recursively extract values from dict/list"""
    if _depth > 20:
        return []
    
    values = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.append(str(key))
            if isinstance(value, (dict, list)):
                values.extend(_extract_values_from_dict(value, _depth + 1))
            elif value is not None:
                values.append(str(value))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                values.extend(_extract_values_from_dict(item, _depth + 1))
            elif item is not None:
                values.append(str(item))
    elif obj is not None:
        values.append(str(obj))
    
    return values

async def inspect_request_flask(req: Request) -> Optional[str]:
    """Inspect request with WAF rules (async)"""
    client_ip = req.client.host
    
    if client_ip in IP_BLACKLIST:
        return "IP_BLACKLIST"

    request_body_str = (await req.body()).decode('utf-8', 'ignore')
    query_string_str = req.url.query or ""
    
    # Parse form data
    try:
        form_data = await req.form()
        all_form_args = {k: v for k, v in form_data.items()}
    except:
        all_form_args = {}
    
    all_query_args = dict(req.query_params)
    
    headers_values = None
    cookies_values = None
    
    for rule in WAF_RULES:
        targets_to_check = set()
        rule_targets = rule.get('target', '').split('|')

        for target_part in rule_targets:
            if target_part in ('URL_PATH', 'REQUEST_URI'):
                targets_to_check.add(req.url.path)
            elif target_part == 'URL_QUERY':
                targets_to_check.add(query_string_str)
            elif target_part == 'HEADERS':
                if headers_values is None:
                    headers_values = set(req.headers.values())
                targets_to_check.update(headers_values)
            elif target_part.startswith('HEADERS:'):
                header_name = target_part.split(':', 1)[1]
                header_val = req.headers.get(header_name, '')
                if header_val:
                    targets_to_check.add(header_val)
            elif target_part == 'BODY':
                targets_to_check.add(request_body_str)
            elif target_part == 'ARGS':
                targets_to_check.update(all_query_args.values())
                targets_to_check.update(all_form_args.values())
            elif target_part == 'COOKIES':
                if cookies_values is None:
                    cookies_values = set(req.cookies.values())
                targets_to_check.update(cookies_values)
            elif target_part == 'REQUEST_METHOD':
                targets_to_check.add(req.method.upper())

        targets_to_check.discard('')
        targets_to_check.discard(None)
        
        if not targets_to_check:
            continue

        operator = rule.get('operator')
        value = rule.get('value')
        action = str(rule.get('action', 'BLOCK')).upper()

        for item in targets_to_check:
            decoded_data, decode_log = deep_decode_data(str(item))

            inspection_targets = [decoded_data]
            if decoded_data:
                parsed_values = parse_structured_data(decoded_data)
                if parsed_values:
                    inspection_targets.extend(parsed_values)

            for target in inspection_targets:
                if evaluate_operator(operator, value, target):
                    if action == 'BLOCK':
                        logger.warning(
                            f"[MATCH][BLOCK] Rule {rule['id']} ('{rule['description']}') "
                            f"triggered on: '{target}'"
                        )
                        return f"RULE_ID: {rule['id']}"

    # ML Layer
    if ML_AVAILABLE and ML_ENABLED:
        ml_result = await check_with_ml(req)
        if ml_result:
            return ml_result
    
    return None

async def check_with_ml(req: Request) -> Optional[str]:
    """ML-based detection (async)"""
    try:
        ml_predictor = get_ml_predictor()
        
        if not ml_predictor.is_loaded:
            logger.info("[ML] Model not loaded, skipping ML check")
            return None
        
        checks = []
        
        url_data = req.url.path or ""
        if req.url.query:
            url_data += "?" + req.url.query
        if url_data and len(url_data.strip()) >= 3:
            checks.append(("URL", url_data))
        
        if req.method in ['POST', 'PUT', 'PATCH']:
            body_data = (await req.body()).decode('utf-8', 'ignore')
            if body_data and len(body_data.strip()) >= 3:
                checks.append(("BODY", body_data))
        
        for check_type, data in checks:
            logger.info(f"[ML] Checking {check_type}: {data[:100]}...")
            
            result = ml_predictor.predict(data)
            if len(result) == 3:
                prediction, confidence, xai_patterns = result
            else:
                prediction, confidence = result
                xai_patterns = {}
            
            if xai_patterns:
                pattern_info = '; '.join([f"{cat}: {', '.join(pats[:3])}" for cat, pats in xai_patterns.items()])
                logger.info(f"[ML][XAI] 🔍 [{check_type}] Detected patterns: {pattern_info}")
            
            logger.info(f"[ML] [{check_type}] Result: {prediction.upper()} | Confidence: {confidence:.4f}")
            
            if prediction == 'attack' and confidence >= ML_CONFIDENCE_THRESHOLD:
                if ML_LIME_ENABLED:
                    ml_predictor.log_explanation(data, num_samples=100)
                
                xai_info = f" | XAI: {list(xai_patterns.keys())}" if xai_patterns else ""
                logger.warning(
                    f"[ML][BLOCK] ⛔ Attack in {check_type}! "
                    f"IP: {req.client.host} | Path: {req.url.path} | "
                    f"Confidence: {confidence:.4f}{xai_info}"
                )
                return f"ML_DETECTION_{check_type}"
            
            if prediction == 'attack' and confidence < ML_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"[ML][ALLOW] ⚠️ [{check_type}] Potential attack but below threshold. "
                    f"Confidence: {confidence:.4f}"
                )
        
        return None
        
    except Exception as e:
        logger.error(f"[ML][ERROR] ML check failed: {e}")
        return None

def is_ip_allowed(client_ip: str, allowed_ips_str: str) -> bool:
    """Check if IP is allowed"""
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    
    for allowed_ip in allowed_ips_str.split(','):
        allowed_ip = allowed_ip.strip()
        if not allowed_ip:
            continue
        try:
            if '/' in allowed_ip:
                if client_ip_obj in ipaddress.ip_network(allowed_ip, strict=False):
                    return True
            else:
                if client_ip_obj == ipaddress.ip_address(allowed_ip):
                    return True
        except ValueError:
            continue
    
    return False

# ====== Routes ======
# NOTE: Specific routes MUST be defined BEFORE the catch-all route

@app.post("/reset-db-management")
async def reset_db_management(request: Request):
    """Reset cache endpoint (async)"""
    allowed_ips = os.getenv("ADMIN_ALLOWED_IPS", "127.0.0.1,192.168.232.1,::1")
    client_ip = request.client.host

    if not is_ip_allowed(client_ip, allowed_ips):
        logger.warning(f"Unauthorized attempt to reset cache from IP: {client_ip}")
        return JSONResponse({"status": "error", "message": "Forbidden"}, status_code=403)
    
    try:
        logger.info("Received API command to reload cache from Admin Panel...")
        await load_cache_from_db()
        logger.info("Cache reloaded successfully via API.")
        return JSONResponse({"status": "success", "message": "Cache reloaded."})
    except Exception as e:
        logger.error(f"Failed to reload cache via API: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/health")
async def health():
    """Health check"""
    return Response(
        content="<html><body>WAF (FastAPI/Async) is running</body></html>", 
        status_code=200, 
        media_type="text/html"
    )

@app.post("/api/explain")
async def explain_request(request: Request):
    """
    On-demand XAI explanation endpoint.
    Generates LIME explanation for a given request payload.
    """
    allowed_ips = os.getenv("ADMIN_ALLOWED_IPS", "127.0.0.1,192.168.232.1,::1,172.18.0.0/16")
    client_ip = request.client.host

    if not is_ip_allowed(client_ip, allowed_ips):
        logger.warning(f"Unauthorized XAI request from IP: {client_ip}")
        return JSONResponse({"status": "error", "message": "Forbidden"}, status_code=403)
    
    try:
        body = await request.json()
        payload = body.get("payload", "")
        num_samples = body.get("num_samples", 200)
        
        if not payload:
            return JSONResponse({"status": "error", "message": "No payload provided"}, status_code=400)
        
        if not ML_AVAILABLE:
            return JSONResponse({"status": "error", "message": "ML not available"}, status_code=503)
        
        ml_predictor = get_ml_predictor()
        if not ml_predictor.is_loaded:
            return JSONResponse({"status": "error", "message": "ML model not loaded"}, status_code=503)
        
        # URL decode payload first for proper analysis
        from urllib.parse import unquote_plus
        decoded_payload = unquote_plus(payload)
        
        logger.info(f"[XAI] Generating LIME explanation for: {decoded_payload[:100]}...")
        
        # Generate LIME explanation on DECODED payload
        explanation = ml_predictor.explain(decoded_payload, num_samples=num_samples)
        
        if explanation is None:
            return JSONResponse({"status": "error", "message": "Failed to generate explanation"}, status_code=500)
        
        logger.info(f"[XAI] Explanation generated - Prediction: {explanation['prediction']}, Confidence: {explanation['confidence']:.2f}")
        
        # Detect patterns for word-level explanation (Option 1)
        from ml_predictor import detect_patterns
        detected_patterns = detect_patterns(decoded_payload)
        
        return JSONResponse({
            "status": "success",
            "explanation": {
                "payload": explanation["payload"],
                "full_payload": decoded_payload,  # Decoded payload for highlighting
                "prediction": explanation["prediction"],
                "p_normal": round(explanation["p_normal"] * 100, 2),
                "p_attack": round(explanation["p_attack"] * 100, 2),
                "confidence": round(explanation["confidence"] * 100, 2),
                "top_dangerous": [{"char": c, "weight": round(w, 4)} for c, w in explanation["top_dangerous"]],
                "top_safe": [{"char": c, "weight": round(w, 4)} for c, w in explanation["top_safe"]],
                "top_dangerous_ngrams": [{"ngram": ng, "weight": round(w, 4)} for ng, w in explanation.get("top_dangerous_ngrams", [])],
                "top_safe_ngrams": [{"ngram": ng, "weight": round(w, 4)} for ng, w in explanation.get("top_safe_ngrams", [])],
                "detected_patterns": detected_patterns  # Word-level patterns
            }
        })
        
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)
    except Exception as e:
        logger.error(f"[XAI] Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Catch-all route for reverse proxy - MUST be last
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def reverse_proxy(request: Request, path: str):
    """Main WAF reverse proxy (async)"""
    block_reason = await inspect_request_flask(request)
    
    if block_reason:
        logger.warning(f"Denied request from IP {request.client.host}. Reason: {block_reason}")
        
        # Parse rule_id and determine action type
        rule_id = None
        action = 'BLOCKED'
        
        if block_reason.startswith("RULE_ID:"):
            # Rule-based block
            try:
                rule_id = int(block_reason.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif block_reason.startswith("ML_DETECTION"):
            # ML-based block
            action = 'ML_BLOCKED'
        elif block_reason == "IP_BLACKLIST":
            action = 'IP_BLOCKED'
        
        await log_event_to_db(request.client.host, request.method, str(request.url), 403, action, rule_id=rule_id)
        return Response(
            content="<html><body><h1>403 Forbidden</h1></body></html>", 
            status_code=403, 
            media_type="text/html"
        )
    
    try:
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('host', None)
        
        # Build backend URL
        backend_url = f"{BACKEND_ADDRESS}/{path}"
        if request.url.query:
            backend_url += f"?{request.url.query}"
        
        # Read request body
        body = await request.body() if request.method in ['POST', 'PUT', 'PATCH'] else None
        
        # Async HTTP client
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                content=body,
                cookies=request.cookies,
                follow_redirects=False
            )
        
        await log_event_to_db(request.client.host, request.method, str(request.url), resp.status_code, 'ALLOWED')
        
        # Copy headers but remove problematic headers to avoid mismatch
        response_headers = dict(resp.headers)
        response_headers.pop('content-length', None)
        response_headers.pop('content-encoding', None)
        response_headers.pop('transfer-encoding', None)
        
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers
        )
        
    except httpx.RequestError as e:
        logger.error(f"Could not connect to backend: {e}")
        await log_event_to_db(request.client.host, request.method, str(request.url), 502, 'ERROR')
        return Response(
            content="<html><body><h1>502 Bad Gateway</h1></body></html>", 
            status_code=502, 
            media_type="text/html"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, workers=4)
