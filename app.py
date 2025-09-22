from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import secrets
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import datetime
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # 必須在 import pyplot 前設定，避免 GUI 後端
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
import pandas as pd
import joblib
import base64
from werkzeug.utils import secure_filename


# 圖片存放路徑
IMAGE_FOLDER = 'static/images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# 中英對照表
FIELD_LABELS = {
    "user_id":"使用者名稱",
    "sex": "性別",
    "age": "年齡",
    "narrow_pitch_range": "音域變窄",
    "decreased_volume": "說話音量變小",
    "fatigue": "說話久了容易累",
    "dryness": "喉嚨常覺得乾",
    "lumping": "喉嚨有異物感",
    "heartburn": "胸口有灼熱感",
    "choking": "吞東西容易嗆到",
    "eye_dryness": "眼睛乾澀",
    "pnd": "鼻涕倒流",
    "diabetes": "糖尿病",
    "hypertension": "高血壓",
    "cad": "心臟病",
    "head_and_neck_cancer": "頭頸部腫瘤",
    "head_injury": "頭部損傷",
    "cva": "腦中風",
    "smoking": "抽菸",
    "ppd": "一天幾包菸",
    "drinking": "喝酒",
    "frequency": "喝酒頻率",
    "onset_of_dysphonia": "症狀如何發生的",
    "noise_at_work": "工作環境是否吵雜",
    "diurnal_pattern": "聲音何時最差",
    "occupational_vocal_demand": "用聲情形",
    "vhi10": "嗓音障礙指標 (VHI-10 總分)",
    # 新增：VHI-10 題目逐題
    "vhi1": "VHI-1 說話時會上氣不接下氣",
    "vhi2": "VHI-2 嗓音一天內變化",
    "vhi3": "VHI-3 大家會問我「你的聲音怎麼了」",
    "vhi4": "VHI-4 聲音聽起來沙啞、乾澀",
    "vhi5": "VHI-5 必須用力才能發聲",
    "vhi6": "VHI-6 清晰度無法預測、變化多端",
    "vhi7": "VHI-7 試著改變我的聲音",
    "vhi8": "VHI-8 說話使我感到吃力",
    "vhi9": "VHI-9 傍晚過後聲音更糟",
    "vhi10_q": "VHI-10 說話說到一半會失控失聲"
}

# 滿滿的對照表(病歷用)
SEX_LABELS = {
    2: "女",
    1: "男"
}
YN_LABELS = {
    0: "無",
    1: "有"
}
SMOKING_LABELS = {
    0: "從未",
    1: "已戒菸",
    2: "有抽菸",
    3: "電子菸"
}
DRINKING1_LABELS = {
    0: "從未",
    1: "已戒酒",
    2: "有喝酒"
}
DRINKING2_LABELS = {
    0: "不喝",
    1: "偶爾喝",
    2: "每周喝",
    3: "幾乎每天"
}
SYMPTOMS_LABELS = {
    1: "突然",
    2: "逐漸變差",
    3: "時好時壞",
    4: "從小開始",
    5: "其他"
}
WORKING_LABELS = {
    1: "否",
    2: "有一點",
    3: "很吵"
}
VOICE1_LABELS = {
    1: "早上",
    2: "下午",
    3: "晚上",
    4: "都一樣",
    5: "不一定"
}
VOICE2_LABELS = {
    1: "總是需要",
    2: "經常需要",
    3: "偶而需要",
    4: "不需要"
}

# 病變模型用的list標準
MODEL_FEATURE_ORDER = [
    "sex",
    "narrow_pitch_range",
    "decreased_volume",
    "fatigue",
    "dryness",
    "choking",
    "eye_dryness",
    "pnd",
    "smoking",
    "drinking",
    "frequency",
    "diurnal_pattern",
    "onset_of_dysphonia",
    "noise_at_work",
    "occupational_vocal_demand",
    "diabetes",
    "hypertension",
    "cad",
    "head_and_neck_cancer",
    "age",
    "vhi10"
]

# 病例模型用的list標準
feature_names = [
        "Sex", "Narrow pitch range", "Decreased volume", "Fatigue", "Dryness",
        "Choking", "Eye dryness", "PND", "Smoking", "Drinking", "Frequency",
        "Diurnal pattern", "Onset of dysphonia", "Noise at work",
        "Occupational vocal demand", "Diabetes", "Hypertension", "CAD",
        "Head and Neck Cancer", "Age", "Voice handicap index - 10"
    ]

# 讀取音訊，製作圖片，回傳檔名
def generate_audio_images(audio_path, filename_base):
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"讀取音訊失敗: {e}")
        return None

    # 波形圖
    plt.figure(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr)
    waveform_file = f"{filename_base}_waveform.png"
    waveform_path = os.path.join(IMAGE_FOLDER, waveform_file)
    plt.savefig(waveform_path)
    plt.close()

    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(6, 2))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    mel_file = f"{filename_base}_mel.png"
    mel_path = os.path.join(IMAGE_FOLDER, mel_file)
    plt.title("Mel-Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(mel_path)
    plt.close()

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    plt.figure(figsize=(6, 2))
    librosa.display.specshow(mfcc, x_axis='time')
    mfcc_file = f"{filename_base}_mfcc.png"
    mfcc_path = os.path.join(IMAGE_FOLDER, mfcc_file)
    plt.title("MFCC")
    plt.colorbar()
    plt.savefig(mfcc_path)
    plt.close()

    return waveform_file, mel_file, mfcc_file

# 判斷副檔名
ALLOWED_EXTENSIONS = {'wav', 'mp3','webm'}
ALLOWED_MIMES = {
    'audio/wav', 'audio/x-wav', 'audio/webm', 'audio/mpeg', 'audio/mp3', 'application/octet-stream'
}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def allowed_mime(mimetype):
    return mimetype in ALLOWED_MIMES

# 資料庫連線用函式
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row  # 查詢結果像 dict 一樣
    return conn

# 建立資料庫
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # 建立使用者表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    """)

    # 建立醫生表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    """)

    # 建立病歷表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS medical_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        sex INTEGER,
        age INTEGER,
        narrow_pitch_range INTEGER,
        decreased_volume INTEGER,
        fatigue INTEGER,
        dryness INTEGER,
        lumping INTEGER,
        heartburn INTEGER,
        choking INTEGER,
        eye_dryness INTEGER,
        pnd INTEGER,
        diabetes INTEGER,
        hypertension INTEGER,
        cad INTEGER,
        head_and_neck_cancer INTEGER,
        head_injury INTEGER,
        cva INTEGER,
        smoking INTEGER,
        ppd INTEGER,
        drinking INTEGER,
        frequency INTEGER,
        onset_of_dysphonia INTEGER,
        noise_at_work INTEGER,
        diurnal_pattern INTEGER,
        occupational_vocal_demand INTEGER,
        -- 新增 VHI-10 題目逐題
        vhi1 INTEGER, vhi2 INTEGER, vhi3 INTEGER, vhi4 INTEGER, vhi5 INTEGER,
        vhi6 INTEGER, vhi7 INTEGER, vhi8 INTEGER, vhi9 INTEGER, vhi10_q INTEGER,
        -- 保留總分
        vhi10 INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 自動遷移：若既有表缺欄位，補上
    cursor.execute("PRAGMA table_info(medical_records)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col in ["vhi1","vhi2","vhi3","vhi4","vhi5","vhi6","vhi7","vhi8","vhi9","vhi10_q"]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE medical_records ADD COLUMN {col} INTEGER DEFAULT 0")

    # 建立分析結果表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        record_id INTEGER,
        result1 TEXT,
        confidence1 REAL,
        result2 TEXT,
        confidence2 REAL,
        audio_blob BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (record_id) REFERENCES medical_records(id)
    )
    """)

    # 新增：分析圖片表（以二進位存 waveform/mel/mfcc）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS result_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER,
        waveform_blob BLOB,
        mel_blob BLOB,
        mfcc_blob BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (result_id) REFERENCES results(id)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized")

# 登入限制
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for('login'))  # 修正：避免導向受保護頁造成無限重導
        return f(*args, **kwargs)
    return decorated_function

# 新增：醫生權限限制
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for('admin'))
        return f(*args, **kwargs)
    return decorated_function

# 整理病歷資料
def organize_records(record):
    if record:
        if "created_at" in record and record["created_at"]:
            utc_time = datetime.datetime.strptime(record["created_at"], "%Y-%m-%d %H:%M:%S")
            local_time = utc_time + datetime.timedelta(hours=8)
            record["created_at"] = local_time.strftime("%Y-%m-%d %H:%M:%S")

    # 安全轉 int，避免從 DB 取回字串導致對照表失效
    def as_int(v):
        try:
            return int(v)
        except Exception:
            return v

    return {
        "user_id": record["username"],
        "sex": SEX_LABELS.get(as_int(record["sex"]), record["sex"]),
        "age": record["age"],
        "narrow_pitch_range": YN_LABELS.get(as_int(record["narrow_pitch_range"]), record["narrow_pitch_range"]),
        "decreased_volume": YN_LABELS.get(as_int(record["decreased_volume"]), record["decreased_volume"]),
        "fatigue": YN_LABELS.get(as_int(record["fatigue"]), record["fatigue"]),
        "dryness": YN_LABELS.get(as_int(record["dryness"]), record["dryness"]),
        "lumping": YN_LABELS.get(as_int(record["lumping"]), record["lumping"]),
        "heartburn": YN_LABELS.get(as_int(record["heartburn"]), record["heartburn"]),
        "choking": YN_LABELS.get(as_int(record["choking"]), record["choking"]),
        "eye_dryness": YN_LABELS.get(as_int(record["eye_dryness"]), record["eye_dryness"]),
        "pnd": YN_LABELS.get(as_int(record["pnd"]), record["pnd"]),
        "diabetes": YN_LABELS.get(as_int(record["diabetes"]), record["diabetes"]),
        "hypertension": YN_LABELS.get(as_int(record["hypertension"]), record["hypertension"]),
        "cad": YN_LABELS.get(as_int(record["cad"]), record["cad"]),
        "head_and_neck_cancer": YN_LABELS.get(as_int(record["head_and_neck_cancer"]), record["head_and_neck_cancer"]),
        "head_injury": YN_LABELS.get(as_int(record["head_injury"]), record["head_injury"]),
        "cva": YN_LABELS.get(as_int(record["cva"]), record["cva"]),
        "smoking": SMOKING_LABELS.get(as_int(record["smoking"]), record["smoking"]),
        "ppd": record["ppd"],
        "drinking": DRINKING1_LABELS.get(as_int(record["drinking"]), record["drinking"]),
        "frequency": DRINKING2_LABELS.get(as_int(record["frequency"]), record["frequency"]),
        "onset_of_dysphonia": SYMPTOMS_LABELS.get(as_int(record["onset_of_dysphonia"]), record["onset_of_dysphonia"]),
        "noise_at_work": WORKING_LABELS.get(as_int(record["noise_at_work"]), record["noise_at_work"]),
        "diurnal_pattern": VOICE1_LABELS.get(as_int(record["diurnal_pattern"]), record["diurnal_pattern"]),
        "occupational_vocal_demand": VOICE2_LABELS.get(as_int(record["occupational_vocal_demand"]), record["occupational_vocal_demand"]),
        # 新增：逐題分數
        "vhi1": record.get("vhi1"),
        "vhi2": record.get("vhi2"),
        "vhi3": record.get("vhi3"),
        "vhi4": record.get("vhi4"),
        "vhi5": record.get("vhi5"),
        "vhi6": record.get("vhi6"),
        "vhi7": record.get("vhi7"),
        "vhi8": record.get("vhi8"),
        "vhi9": record.get("vhi9"),
        "vhi10_q": record.get("vhi10_q"),
        "vhi10": record["vhi10"],
        "created_at": record["created_at"]
    }

# 病例分析模型
def predict_from_list(input_list, model_path, feature_names):
    df = pd.DataFrame([input_list], columns=feature_names)
    model = joblib.load(model_path)
    y_pred = model.predict(df)
    y_pred_proba = model.predict_proba(df)
    pred_label = y_pred[0]
    try:
        pred_label_scalar = int(pred_label)
    except Exception:
        pred_label_scalar = pred_label
    classes = getattr(model, "classes_", None)
    return pred_label_scalar, y_pred_proba, classes  # 多回傳 classes_ 便於對應置信度

# 嗓音模型
def predict_audio_file(file_path, model_path, class_names, sr=22050, duration=3.0, n_mfcc=40, max_mfcc_length=130):
    # 讀取音訊
    signal, sr = librosa.load(file_path, sr=sr)
    total_duration = librosa.get_duration(y=signal, sr=sr)
    # 擷取最中間三秒
    if total_duration > duration:
        center = len(signal) // 2
        half_len = int(sr * duration / 2)
        start = max(center - half_len, 0)
        end = min(center + half_len, len(signal))
        signal = signal[start:end]
    # 計算 MFCC 並修正長度
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] > max_mfcc_length:
        mfcc = mfcc[:, :max_mfcc_length]
    elif mfcc.shape[1] < max_mfcc_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_mfcc_length - mfcc.shape[1])), mode='constant')
    mfcc = mfcc[..., np.newaxis]
    X = np.array([mfcc])
    # 載入模型并預測
    model = load_model(model_path)
    y_pred = model.predict(X)
    y_pred_class = np.argmax(y_pred, axis=1)
    pred_label = class_names[y_pred_class[0]]
    pred_proba = y_pred[0]
    return pred_label, pred_proba

app = Flask(__name__) # 建立Application物件
# 使用環境變數 SECRET_KEY，無則退回隨機
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

UPLOAD_FOLDER = 'uploads/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 安全強化：限制上傳大小（例：10MB）
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
# Session/Cookie 安全
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False  # 部署 HTTPS 後請改為 True
)

# 安全標頭
@app.after_request
def set_security_headers(resp):
    # 注意：目前使用內嵌 <script>/<style> 與 data URI，因此 CSP 暫放寬；未來可移除 inline 並收緊
    resp.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; media-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline';"
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['Permissions-Policy'] = 'microphone=(), camera=()'
    return resp

# 網站首頁
@app.route('/')
def index():
    return render_template('index.html')

# 註冊
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 基礎密碼強度檢查（長度>=8、含字母與數字）
        if len(password) < 8 or password.isalpha() or password.isdigit():
            return render_template("register.html", error="密碼需至少8碼，且包含字母與數字")

        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template("register.html",error="帳號已存在")
        finally:
            conn.close()

        return redirect(url_for('login'))
    return render_template('register.html')

# 登入
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = False  # 一般使用者
            return redirect(url_for('Main'))
        else:
            return render_template("login.html", error="帳號或密碼錯誤")
    return render_template('login.html')

# 登出
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# 醫生註冊
@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if len(password) < 8 or password.isalpha() or password.isdigit():
            return render_template("admin_register.html", error="密碼需至少8碼，且包含字母與數字")

        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template("admin_register.html",error="帳號已存在")
        finally:
            conn.close()

        return redirect(url_for('admin'))
    return render_template('admin_register.html')

# 醫生登入
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        admin = conn.execute("SELECT * FROM admins WHERE username=?", (username,)).fetchone()
        conn.close()
        # 移除印出帳密的 debug 紀錄
        if admin and check_password_hash(admin['password'], password):
            # 不覆蓋一般使用者的 user_id
            session.clear()
            session['admin_id'] = admin['id']
            session['username'] = admin['username']
            session['is_admin'] = True
            return redirect(url_for('admin_Main'))
        else:
            return render_template("admin.html", error="帳號或密碼錯誤")
    return render_template('admin.html')

# 醫生功能頁面
@app.route('/admin_Main')
@admin_required
def admin_Main():
    q = request.args.get('q', '').strip()
    user_id = request.args.get('user_id', '').strip()

    matches = []
    selected_user = None
    results = []
    record_pretty = None  # 新增：最新病歷摘要

    conn = get_db_connection()
    try:
        if q:
            matches = conn.execute(
                "SELECT id, username FROM users WHERE username LIKE ? ORDER BY username LIMIT 50",
                (f"%{q}%",)
            ).fetchall()

        if user_id:
            try:
                uid = int(user_id)
            except ValueError:
                uid = None
            if uid:
                selected_user = conn.execute(
                    "SELECT id, username FROM users WHERE id = ?",
                    (uid,)
                ).fetchone()

                # 新增：撈最新病歷並美化
                rec = conn.execute("""
                    SELECT medical_records.*, users.username
                    FROM medical_records
                    JOIN users ON medical_records.user_id = users.id
                    WHERE medical_records.user_id = ?
                    ORDER BY medical_records.created_at DESC LIMIT 1
                """, (uid,)).fetchone()
                if rec:
                    record_pretty = organize_records(dict(rec))

                # 原有：撈該使用者歷史檢查紀錄
                rows = conn.execute(
                    "SELECT * FROM results WHERE user_id = ? ORDER BY created_at DESC",
                    (uid,)
                ).fetchall()
                for r in rows:
                    rd = dict(r)
                    if rd.get('audio_blob'):
                        rd['audio_src'] = "data:audio/wav;base64," + base64.b64encode(rd['audio_blob']).decode('utf-8')

                    # 讀取圖片 BLOB
                    img = conn.execute(
                        "SELECT waveform_blob, mel_blob, mfcc_blob FROM result_images WHERE result_id = ?",
                        (rd['id'],)
                    ).fetchone()
                    if img:
                        wf, mel, mfcc = img['waveform_blob'], img['mel_blob'], img['mfcc_blob']
                        if wf:
                            rd['waveform_src'] = "data:image/png;base64," + base64.b64encode(wf).decode('utf-8')
                        if mel:
                            # 修正：去掉多餘的括號
                            rd['mel_src'] = "data:image/png;base64," + base64.b64encode(mel).decode('utf-8')
                        if mfcc:
                            rd['mfcc_src'] = "data:image/png;base64," + base64.b64encode(mfcc).decode('utf-8')

                    # 新增：為每筆結果找「檢測時間以前最近」的一筆病歷摘要
                    rec_at_time = conn.execute("""
                        SELECT mr.*, u.username
                        FROM medical_records mr
                        JOIN users u ON mr.user_id = u.id
                        WHERE mr.user_id = ? AND mr.created_at <= ?
                        ORDER BY mr.created_at DESC
                        LIMIT 1
                    """, (uid, rd['created_at'])).fetchone()
                    if rec_at_time:
                        rd['record'] = organize_records(dict(rec_at_time))

                    results.append(rd)
    finally:
        conn.close()

    matches = [dict(m) for m in matches]
    selected_user = dict(selected_user) if selected_user else None

    return render_template(
        "admin_Main.html",
        q=q,
        matches=matches,
        selected_user=selected_user,
        results=results,
        record=record_pretty,              # 新增：病歷摘要
        field_labels=FIELD_LABELS          # 新增：欄位對照
    )

# 主要功能頁面
@app.route('/Main')
@login_required # 限制訪問
def Main():
    # 讀最新病歷
    conn = get_db_connection()
    record = conn.execute("""
    SELECT medical_records.*, users.username
    FROM medical_records
    JOIN users ON medical_records.user_id = users.id
    WHERE username = ?
    ORDER BY medical_records.created_at DESC LIMIT 1""",(session['username'],)).fetchone()
    conn.close()

    record = dict(record) if record else None
    record = organize_records(record) if record else None

    return render_template("Main.html", record=record, field_labels=FIELD_LABELS)

# 上傳音訊檔案
@app.route('/upload_audio', methods=['POST'])
@login_required
def upload_audio():
    file = request.files.get('audio_file')
    if not file:
        return jsonify({"error": "未接收到檔案"}), 400
    if not allowed_file(file.filename) or not allowed_mime(file.mimetype):
        return jsonify({"error": "僅允許上傳 .wav/.mp3/.webm 且需為有效音訊 MIME 類型"}), 400

    # 以時間戳＋隨機字串命名，避免原檔名風險
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{secrets.token_hex(4)}.{ext}")

    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"上傳失敗：{e}"}), 500

    return jsonify({"audio_path": save_path})

# 上傳錄音檔案
@app.route('/record_audio', methods=['POST'])
@login_required
def record_audio():
    file = request.files.get('audio_data')
    if not file:
        return jsonify({"error": "未接收到錄音"}), 400
    if not allowed_mime(file.mimetype):
        return jsonify({"error": "錄音格式不被接受"}), 400

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{secrets.token_hex(4)}.webm")
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"錄音儲存失敗：{e}"}), 500

    return jsonify({"audio_path": save_path})

# 手動輸入
@app.route('/input_medical')
@login_required # 限制訪問
def input_medical():
    return render_template('input_medical.html')

# 存病歷資料
@app.route('/save_medical', methods=['POST'])
@login_required
def save_medical():
    data = request.form
    # 註：前端 vhi10 的最後一題 select 名稱是 vhi10（非 vhi10_q）
    vhi_items = [int(data.get(f'vhi{i}', 0)) for i in range(1, 11)]
    vhi10_total = sum(vhi_items)

    # 入庫前統一轉 int
    int_fields = [
        'sex','age','narrow_pitch_range','decreased_volume','fatigue','dryness','lumping','heartburn',
        'choking','eye_dryness','pnd','diabetes','hypertension','cad','head_and_neck_cancer','head_injury',
        'cva','smoking','ppd','drinking','frequency','onset_of_dysphonia','noise_at_work',
        'diurnal_pattern','occupational_vocal_demand'
    ]
    val = {k: int(data[k]) for k in int_fields}

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO medical_records (
            user_id, sex, age, narrow_pitch_range, decreased_volume,
            fatigue, dryness, lumping, heartburn, choking, eye_dryness, pnd,
            diabetes, hypertension, cad, head_and_neck_cancer, head_injury, cva,
            smoking, ppd, drinking, frequency, onset_of_dysphonia, noise_at_work,
            diurnal_pattern, occupational_vocal_demand,
            vhi1, vhi2, vhi3, vhi4, vhi5, vhi6, vhi7, vhi8, vhi9, vhi10_q,
            vhi10
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?)
    """, (
        session['user_id'],
        val['sex'], val['age'], val['narrow_pitch_range'], val['decreased_volume'],
        val['fatigue'], val['dryness'], val['lumping'], val['heartburn'], val['choking'],
        val['eye_dryness'], val['pnd'], val['diabetes'], val['hypertension'], val['cad'],
        val['head_and_neck_cancer'], val['head_injury'], val['cva'], val['smoking'], val['ppd'],
        val['drinking'], val['frequency'], val['onset_of_dysphonia'], val['noise_at_work'],
        val['diurnal_pattern'], val['occupational_vocal_demand'],
        # 逐題（注意：vhi10_q 對應前端的 vhi10）
        vhi_items[0], vhi_items[1], vhi_items[2], vhi_items[3], vhi_items[4],
        vhi_items[5], vhi_items[6], vhi_items[7], vhi_items[8], vhi_items[9],
        vhi10_total
    ))
    conn.commit()
    conn.close()

    return redirect(url_for('Main'))

#分析結果歷史紀錄
@app.route('/medical')
@login_required#限制訪問
def medical():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT * FROM results WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    results = [] 
    for r in rows: 
        row_dict = dict(r) 
        if row_dict.get('audio_blob'): 
            b64_audio = base64.b64encode(row_dict['audio_blob']).decode('utf-8') 
            row_dict['audio_src'] = f"data:audio/wav;base64,{b64_audio}" 
        results.append(row_dict) 
 
    return render_template('medical.html', results=results)

#確認後產生圖片顯示圖片
@app.route('/confirm_analysis', methods=['POST'])
@login_required
def confirm_analysis():
    audio_path = request.form.get("audio_path")
    # 先檢查音訊是否存在
    if not audio_path or not os.path.exists(audio_path):
        # 撈最新病歷供頁面顯示
        conn = get_db_connection()
        record = conn.execute("""
        SELECT medical_records.*, users.username
        FROM medical_records
        JOIN users ON medical_records.user_id = users.id
        WHERE username = ?
        ORDER BY medical_records.created_at DESC LIMIT 1""", (session['username'],)).fetchone()
        conn.close()
        record = dict(record) if record else None
        record = organize_records(record) if record else None
        return render_template("Main.html",
                               error="沒有找到音訊，請重新錄音或上傳",
                               record=record,
                               field_labels=FIELD_LABELS)

    # 撈最新病歷（後續分析也需要）
    conn = get_db_connection()
    record = conn.execute("""
    SELECT medical_records.*, users.username
    FROM medical_records
    JOIN users ON medical_records.user_id = users.id
    WHERE username = ?
    ORDER BY medical_records.created_at DESC LIMIT 1""", (session['username'],)).fetchone()
    conn.close()

    if not record:
        return render_template("Main.html",
                               error="請先新增病歷後再分析",
                               record=None,
                               field_labels=FIELD_LABELS)

    record = dict(record)
    # 建立模型輸入時先轉 int，避免字串型別造成模型報錯或行為不一致
    def as_int(v):
        try:
            return int(v)
        except Exception:
            return v
    sample_list = [as_int(record[col]) for col in MODEL_FEATURE_ORDER]
    record_id = record["id"]
    record_user = record["user_id"]
    record_pretty = organize_records(record)

    # 病例分析模型：回傳 pred1、proba 與 classes_，以正確對應置信度
    model_path1 = 'best_svm_model.pkl'
    pred1, proba1, classes1 = predict_from_list(sample_list, model_path1, feature_names)
    type1, conf1 = None, None
    if pred1 == 1:
        type1 = "結構性病變"
    elif pred1 == 2:
        type1 = "非結構性病變"
    elif pred1 == 3:
        type1 = "健康"

    # 依 classes_ 找到 pred1 對應的索引，避免取錯機率
    if classes1 is not None:
        classes_list = list(classes1)
        try:
            idx = classes_list.index(pred1)
        except ValueError:
            idx = int(np.argmax(proba1[0]))
        conf1 = float(proba1[0][idx])
    else:
        conf1 = float(np.max(proba1[0]))

    # 結構性病變嗓音模型
    type2, conf2 = None, None
    if pred1 == 1:
        model_path2 = 'smote_cnn_model_0.86.h5'
        class_names = [
            '1.Polyp & 2.Nodules', '13.Ulcer', '15.Dysplasia', '16.Laryngeal cancer', '17.Papilloma',
            '19.Scar', '20.Varix', '5.Vocal paresis', '6.Vocal palsy', "7.Reinke's edema",
            '8.Sulcus', '11.Vocal process granuloma', '12.Fibrous mass'
        ]
        chinese_name = ['聲帶息肉 & 聲帶結節','聲帶潰瘍','表皮異常增生','喉癌','乳突瘤',
                        '聲帶疤痕','微血管增生','單側聲帶輕癱','單側聲帶麻痹','慢性聲帶水腫',
                        '聲帶溝','聲帶突肉芽腫','纖維斑塊']
        pred2, proba2 = predict_audio_file(audio_path, model_path2, class_names)
        for i, cname in enumerate(class_names):
            if pred2 == cname:
                type2 = chinese_name[i]
                conf2 = float(proba2[i])
                break

    # 非結構性病變嗓音模型
    if pred1 == 2:
        model_path2 = 'smote_non_cnn_model_0.84.h5'
        class_names = ['9.Muscle tension dysphonia','10.Presbyphonia','14.Spasmodic dysphonia','18.Tremor']
        pred2, proba2 = predict_audio_file(audio_path, model_path2, class_names)
        if pred2 == '9.Muscle tension dysphonia':
            type2, conf2 = "肌肉緊張性發聲異常", float(proba2[0])
        elif pred2 == '10.Presbyphonia':
            type2, conf2 = "嗓音老化", float(proba2[1])
        elif pred2 == '14.Spasmodic dysphonia':
            type2, conf2 = "聲帶痙攣", float(proba2[2])
        elif pred2 == '18.Tremor':
            type2, conf2 = "聲帶顫抖", float(proba2[3])

    # 信心程度低的提示旗標
    type3 = 1 if ((conf1 is not None and conf1 < 0.8) or (conf2 is not None and conf2 < 0.8)) else None

    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result = generate_audio_images(audio_path, timestamp)
        if result is None:
            return render_template("Main.html",
                                   error="音訊讀取錯誤，請確認檔案格式",
                                   record=record_pretty,
                                   field_labels=FIELD_LABELS)
        waveform_img, mel_img, mfcc_img = result

        # 存結果 + 讀入音檔 BLOB
        with open(audio_path, "rb") as f:
            audio_blob = f.read()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO results (user_id, record_id, result1, confidence1, result2, confidence2, audio_blob)
            VALUES (?,?,?,?,?,?,?)
        """, (
            record_user,
            record_id,
            type1,
            float(conf1) if conf1 is not None else None,
            type2,
            float(conf2) if conf2 is not None else None,
            audio_blob
        ))
        result_id = cur.lastrowid  # 取得剛插入的結果ID

        # 讀取三張分析 PNG 檔，寫入 result_images
        with open(os.path.join(IMAGE_FOLDER, waveform_img), "rb") as f:
            waveform_blob = f.read()
        with open(os.path.join(IMAGE_FOLDER, mel_img), "rb") as f:
            mel_blob = f.read()
        with open(os.path.join(IMAGE_FOLDER, mfcc_img), "rb") as f:
            mfcc_blob = f.read()

        cur.execute("""
            INSERT INTO result_images (result_id, waveform_blob, mel_blob, mfcc_blob)
            VALUES (?,?,?,?)
        """, (result_id, waveform_blob, mel_blob, mfcc_blob))

        conn.commit()
        conn.close()

        return render_template("Main.html",
                               waveform_img=waveform_img,
                               mel_img=mel_img,
                               mfcc_img=mfcc_img,
                               record=record_pretty,
                               type1=type1,
                               conf1=conf1,
                               type2=type2,
                               conf2=conf2,
                               type3=type3,
                               field_labels=FIELD_LABELS)
    except Exception as e:
        return render_template("Main.html",
                               error=f"分析失敗：{e}",
                               record=record_pretty,
                               field_labels=FIELD_LABELS)

if __name__ == '__main__':
    init_db()#啟動時確保資料表存在
    app.run(host='0.0.0.0',debug=True)