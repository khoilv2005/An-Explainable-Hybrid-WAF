import sys
import os
import requests
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify

# Thêm thư mục gốc của dự án vào Python Path để có thể import từ 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các thành phần ORM từ file dùng chung
try:
    from shared.database import SessionLocal, Rule, IPBlacklist, ActivityLog, logger, init_database
except ImportError:
    print("FATAL ERROR: Could not import from 'shared/database.py'.")
    print("Please ensure the file exists and the project structure is correct.")
    sys.exit(1)

# Load configuration from environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Cấu hình ---
WAF_RESET_URL = os.getenv("WAF_RESET_URL", "http://127.0.0.1:8080/reset-db-management")
WAF_API_URL = os.getenv("WAF_API_URL", "http://waf_app:8080")  # WAF container URL
LISTEN_HOST = os.getenv("ADMIN_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("ADMIN_LISTEN_PORT", "5000"))

app = Flask(__name__)
app.secret_key = os.getenv("ADMIN_SECRET_KEY", "super_secret_key_123") 
# --- Các hàm tiện ích ---
def notify_waf_to_reset():
    """Gửi một request đến WAF để yêu cầu nó tải lại rule."""
    try:
        response = requests.post(WAF_RESET_URL, timeout=5)
        if response.status_code == 200:
            print("[INFO][Admin] Notified WAF to reload rules successfully.")
        else:
            print(f"[WARNING][Admin] Failed to notify WAF. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR][Admin] Could not connect to WAF to send reset command: {e}")

# --- API Routes ---

@app.route('/api/explain', methods=['POST'])
def api_explain():
    """
    Proxy endpoint to get LIME XAI explanation from WAF.
    Forwards request to WAF's /api/explain endpoint.
    """
    try:
        data = request.get_json()
        if not data or 'payload' not in data:
            return jsonify({'status': 'error', 'message': 'No payload provided'}), 400
        
        # Forward to WAF
        waf_url = f"{WAF_API_URL}/api/explain"
        response = requests.post(
            waf_url,
            json=data,
            timeout=30  # LIME can take time
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'message': 'XAI request timed out (LIME is slow)'}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get XAI explanation: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to WAF: {str(e)}'}), 502
    except Exception as e:
        logger.error(f"XAI API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Page Routes ---


# Route cho Dashboard Chính
@app.route('/')
def admin_dashboard():
    session = SessionLocal()
    try:
        total_requests = session.query(ActivityLog).count()
        blocked_requests = session.query(ActivityLog).filter(ActivityLog.action_taken.in_(['BLOCKED', 'ML_BLOCKED', 'IP_BLOCKED', 'ERROR'])).count()
        allowed_requests = session.query(ActivityLog).filter_by(action_taken='ALLOWED').count()
        active_rules = session.query(Rule).filter_by(enabled=True).count()
        blacklisted_ips = session.query(IPBlacklist).count()

        # Thống kê theo category
        from sqlalchemy import func
        category_stats = session.query(
            Rule.category,
            func.count(Rule.id).label('rule_count')
        ).filter_by(enabled=True).group_by(Rule.category).all()

        stats = {
            'total_requests': total_requests,
            'blocked_requests': blocked_requests,
            'allowed_requests': allowed_requests,
            'active_rules': active_rules,
            'blacklisted_ips': blacklisted_ips,
            'category_stats': [{'category': cat, 'count': count} for cat, count in category_stats]
        }

        # Get pagination parameters for initial load
        page = int(request.args.get('page', 1))
        per_page = 100
        offset = (page - 1) * per_page

        logs = session.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).offset(offset).limit(per_page).all()

        return render_template('admin_dashboard.html', stats=stats, logs=logs)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Database connection error. Please try again.', 'error')
        return render_template('admin_dashboard.html', stats={}, logs=[])
    finally:
        session.close()

# API Route cho latest logs (AJAX endpoint) với pagination
@app.route('/api/logs/latest')
def api_logs_latest():
    session = SessionLocal()
    try:
        # Get pagination and filter parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))
        action_filter = request.args.get('action', 'ALL')  # ALL, ALLOWED, or BLOCKED

        # Build base query
        base_query = session.query(ActivityLog)

        # Apply action filter if specified
        if action_filter == 'ALLOWED':
            base_query = base_query.filter(ActivityLog.action_taken == 'ALLOWED')
            logger.info(f"Applied filter: {action_filter}")
        elif action_filter == 'BLOCKED':
            base_query = base_query.filter(ActivityLog.action_taken.in_(['BLOCKED', 'ML_BLOCKED', 'IP_BLOCKED', 'ERROR']))
            logger.info(f"Applied filter: {action_filter} (includes BLOCKED, ML_BLOCKED, IP_BLOCKED and ERROR)")

        # Get total count for pagination (with filter applied)
        total_logs = base_query.count()
        logger.info(f"Total logs with filter {action_filter}: {total_logs}")

        # Tính offset cho phân trang chuẩn
        if page < 1:
            page = 1
        offset = (page - 1) * per_page

        logs = (
            base_query
            .order_by(ActivityLog.timestamp.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )
        logger.info(f"Retrieved {len(logs)} logs (latest first) for page {page} (offset={offset})")

        # Get current statistics
        all_total_requests = session.query(ActivityLog).count()
        all_blocked_requests = session.query(ActivityLog).filter(ActivityLog.action_taken.in_(['BLOCKED', 'ML_BLOCKED', 'IP_BLOCKED', 'ERROR'])).count()
        all_allowed_requests = session.query(ActivityLog).filter_by(action_taken='ALLOWED').count()
        active_rules = session.query(Rule).filter_by(enabled=True).count()
        blacklisted_ips = session.query(IPBlacklist).count()

        # Convert logs to JSON format
        logs_data = []
        for log in logs:
            logs_data.append({
                'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S') if log.timestamp else '',
                'client_ip': log.client_ip or '',
                'request_method': log.request_method or '',
                'request_path': log.request_path or '',
                'action_taken': log.action_taken or '',
                'status_code': log.status_code or '',
                'triggered_rule_id': log.triggered_rule_id
            })

        # Statistics data - always return overall statistics, not filtered
        stats_data = {
            'total_requests': all_total_requests,
            'blocked_requests': all_blocked_requests,
            'allowed_requests': all_allowed_requests,
            'active_rules': active_rules,
            'blacklisted_ips': blacklisted_ips
        }

        # Pagination info
        total_pages = (total_logs + per_page - 1) // per_page if total_logs > 0 else 1
        pagination_info = {
            'current_page': page,
            'per_page': per_page,
            'total_logs': total_logs,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }

        return jsonify({
            'logs': logs_data,
            'stats': stats_data,
            'pagination': pagination_info
        })

    except Exception as e:
        logger.error(f"API logs latest error: {e}")
        return jsonify({'error': 'Database error', 'logs': [], 'stats': {}, 'pagination': {}}), 500
    finally:
        session.close()

# Route để XEM danh sách rule
@app.route('/manage-rules')
def manage_rules():
    session = SessionLocal()
    try:
        rules = session.query(Rule).order_by(Rule.id).all()
        return render_template('manage_rules.html', rules=rules)
    except Exception as e:
        logger.error(f"Manage rules error: {e}")
        flash('Database connection error. Please try again.', 'error')
        return render_template('manage_rules.html', rules=[])
    finally:
        session.close()

# Route để THÊM rule mới
@app.route('/add-rule', methods=['POST'])
def add_rule():
    session = SessionLocal()
    try:
        # Chuẩn hóa action từ form
        raw_action = request.form.get('rule_action', 'BLOCK')
        action = str(raw_action).upper()
        if action not in ('BLOCK', 'LOG', 'ALLOW'):
            action = 'BLOCK'

        new_rule = Rule(
            id=int(request.form['id']),
            description=request.form['description'],
            category=request.form['category'],  # Thêm trường category mới
            target=request.form['target'],
            operator=request.form['operator'],
            value=request.form['value'],
            enabled='enabled' in request.form,
            severity='MEDIUM', # Gán giá trị mặc định
            action=action
        )
        session.add(new_rule)
        session.commit()
        
        # Gửi tín hiệu cho WAF sau khi commit thành công
        notify_waf_to_reset()
        flash('Rule added successfully!', 'success')
    except Exception as e:
        logger.error(f"Could not add rule: {e}")
        flash('Failed to add rule. Please try again.', 'error')
        session.rollback()
    finally:
        session.close()
    
    return redirect(url_for('manage_rules'))

# Route để XÓA một rule
@app.route('/delete-rule/<int:rule_id>', methods=['POST'])
def delete_rule(rule_id):
    session = SessionLocal()
    try:
        rule_to_delete = session.query(Rule).get(rule_id)
        if rule_to_delete:
            session.delete(rule_to_delete)
            session.commit()
            flash('Rule deleted successfully!', 'success')
            # Gửi tín hiệu cho WAF sau khi commit thành công
            notify_waf_to_reset()
        else:
            logger.warning(f"Rule with ID {rule_id} not found for deletion.")
            flash('Rule not found.', 'warning')
    except Exception as e:
        logger.error(f"Could not delete rule {rule_id}: {e}")
        flash('Failed to delete rule. Please try again.', 'error')
        session.rollback()
    finally:
        session.close()

    return redirect(url_for('manage_rules'))


@app.route('/delete-all-rules', methods=['POST'])
def delete_all_rules():
    """
    Xóa toàn bộ rules hiện tại, giữ nguyên logs và blacklist.
    Sau đó yêu cầu WAF reload lại cache rules từ DB.
    """
    session = SessionLocal()
    try:
        deleted_rules = session.query(Rule).delete()
        session.commit()

        flash(f'Deleted {deleted_rules} rules from database.', 'success')

        # Cho WAF reload lại rules từ DB
        notify_waf_to_reset()
    except Exception as e:
        logger.error(f"Could not delete all rules: {e}")
        flash('Failed to delete all rules. Please check logs.', 'error')
        session.rollback()
    finally:
        session.close()

    return redirect(url_for('manage_rules'))

@app.route('/import-rules', methods=['POST'])
def import_rules():
    # 1. Kiểm tra file upload
    if 'rule_file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('manage_rules'))
    
    file = request.files['rule_file']
    if file.filename == '':
        flash('No file selected for uploading.', 'error')
        return redirect(url_for('manage_rules'))

    if file and file.filename.endswith('.json'):
        session = SessionLocal()
        try:
            # 2. Đọc và phân tích file JSON
            content = file.read().decode('utf-8')
            
            # Kiểm tra nội dung file không rỗng
            if not content.strip():
                flash('The uploaded file is empty.', 'error')
                return redirect(url_for('manage_rules'))
            
            rules_from_json = json.loads(content)
            
            if not isinstance(rules_from_json, list):
                raise ValueError("JSON content must be a list of rules.")

            added_count = 0
            skipped_count = 0

            # 3. Thêm rule vào DB
            for rule_data in rules_from_json:
                # Validate required fields
                required_fields = ['id', 'description', 'target', 'operator', 'value']
                missing_fields = [field for field in required_fields if not rule_data.get(field)]
                if missing_fields:
                    flash(f'Rule with ID {rule_data.get("id", "unknown")} is missing required fields: {", ".join(missing_fields)}', 'error')
                    continue

                # Chuẩn hóa và validate action
                raw_action = rule_data.get('action', 'BLOCK')
                action = str(raw_action).upper()
                allowed_actions = {'BLOCK', 'LOG', 'ALLOW'}
                if action not in allowed_actions:
                    # Nếu giá trị không hợp lệ, fallback về BLOCK và ghi log cảnh báo
                    logger.warning(
                        f"Rule ID {rule_data.get('id')} has invalid action '{raw_action}', "
                        "falling back to 'BLOCK'."
                    )
                    action = 'BLOCK'

                # Kiểm tra xem rule ID đã tồn tại chưa
                existing_rule = session.query(Rule).filter_by(id=rule_data.get('id')).first()
                if existing_rule:
                    skipped_count += 1
                    continue # Bỏ qua nếu đã có
                
                new_rule = Rule(
                    id=rule_data.get('id'),
                    enabled=rule_data.get('enabled', True),
                    description=rule_data.get('description'),
                    category=rule_data.get('category', 'Custom'),  # Thêm trường category với default
                    severity=rule_data.get('severity', 'MEDIUM'),
                    target=rule_data.get('target'),
                    operator=rule_data.get('operator'),
                    value=rule_data.get('value'),
                    action=action
                )
                session.add(new_rule)
                session.flush()  # Flush để phát hiện lỗi duplicate ngay lập tức
                added_count += 1
            
            session.commit()
            flash(f'Successfully imported {added_count} new rules. Skipped {skipped_count} existing rules.', 'success')
            
            # 4. Gửi tín hiệu cho WAF
            if added_count > 0:
                notify_waf_to_reset()

        except json.JSONDecodeError as e:
            flash('Invalid JSON format in the uploaded file.', 'error')
            logger.error(f"JSON decode error: {e}")
            session.rollback()
        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            logger.error(f"Import rules error: {e}")
            session.rollback()
        finally:
            session.close()
    else:
        flash('Invalid file type. Please upload a .json file.', 'error')

    return redirect(url_for('manage_rules'))


# Route để hiển thị danh sách IP trong blacklist
@app.route('/blacklist')
def view_blacklist():
    session = SessionLocal()
    try:
        # Lấy danh sách IP trong blacklist với thông tin chi tiết
        blacklisted_ips = session.query(IPBlacklist).order_by(IPBlacklist.timestamp.desc()).all()

        # Thống kê thêm thông tin
        total_ips = len(blacklisted_ips)

        # Lấy thông tin thống kê về các IP bị chặn
        ip_stats = []
        for ip in blacklisted_ips:
            # Đếm số lần vi phạm của IP này
            violation_count = session.query(ActivityLog).filter_by(
                client_ip=ip.ip_address,
                action_taken='BLOCKED'
            ).count()

            ip_stats.append({
                'ip_address': ip.ip_address,
                'blacklisted_at': ip.timestamp,
                'triggered_rule_id': ip.triggered_rule_id,
                'notes': ip.notes,
                'violation_count': violation_count
            })

    except Exception as e:
        logger.error(f"Could not load blacklist: {e}")
        flash('Failed to load blacklist data.', 'error')
        ip_stats = []
        total_ips = 0
    finally:
        session.close()

    return render_template('blacklist.html',
                         blacklisted_ips=ip_stats,
                         total_ips=total_ips)


# Route để xóa IP khỏi blacklist
@app.route('/remove-from-blacklist/<string:ip_address>', methods=['POST'])
def remove_from_blacklist(ip_address):
    session = SessionLocal()
    try:
        ip_to_remove = session.query(IPBlacklist).filter_by(ip_address=ip_address).first()
        if ip_to_remove:
            # Xóa IP khỏi blacklist
            session.delete(ip_to_remove)

            # Xóa tất cả các logs BLOCKED của IP này để reset violation count
            deleted_logs = session.query(ActivityLog).filter_by(
                client_ip=ip_address,
                action_taken='BLOCKED'
            ).delete()

            session.commit()
            flash(f'IP {ip_address} has been removed from blacklist and cleared {deleted_logs} violation logs.', 'success')

            # Gửi tín hiệu cho WAF để reload cache
            notify_waf_to_reset()
        else:
            flash(f'IP {ip_address} not found in blacklist.', 'warning')
    except Exception as e:
        logger.error(f"Could not remove IP {ip_address} from blacklist: {e}")
        flash('Failed to remove IP from blacklist.', 'error')
        session.rollback()
    finally:
        session.close()

    return redirect(url_for('view_blacklist'))


@app.route('/reset-all-data', methods=['POST'])
def reset_all_data():
    """
    Xóa toàn bộ dữ liệu WAF: Rules, ActivityLog, IPBlacklist
    và yêu cầu WAF reload cache từ DB trống.
    """
    session = SessionLocal()
    try:
        # Xóa log hoạt động trước (để tránh FK constraint nếu có)
        deleted_logs = session.query(ActivityLog).delete()
        deleted_blacklist = session.query(IPBlacklist).delete()
        deleted_rules = session.query(Rule).delete()
        session.commit()

        flash(
            f'Reset database successfully: '
            f'{deleted_rules} rules, {deleted_blacklist} blacklisted IPs, {deleted_logs} logs deleted.',
            'success'
        )

        # Thông báo cho WAF reload cache (sẽ load DB trống)
        notify_waf_to_reset()
    except Exception as e:
        logger.error(f"Could not reset all WAF data: {e}")
        flash('Failed to reset database. Please check logs.', 'error')
        session.rollback()
    finally:
        session.close()

    return redirect(url_for('manage_rules'))

# --- Main ---
if __name__ == "__main__":
    # Initialize database first
    if not init_database():
        logger.error("Failed to initialize database. Exiting...")
        sys.exit(1)

    logger.info(f"Admin Panel is running on http://{LISTEN_HOST}:{LISTEN_PORT}")
    app.run(host=LISTEN_HOST, port=LISTEN_PORT)