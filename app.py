from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import secrets
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import datetime
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
import pandas as pd
import joblib
matplotlib.use('Agg')# Flask是用非 GUI 的背景線程執行，調整為非GUI後端，否則連續上傳圖片會有GUI殘留(會報錯)

# 圖片存放路徑
IMAGE_FOLDER = 'static/images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

#中英對照表
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
    "created_at": "建立時間(UTC+8)"
}

#滿滿的對照表(病歷用)
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

#病變模型用的list標準
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

#病例模型用的list標準
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
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        vhi10 INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

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

    conn.commit()
    conn.close()
    print("✅ Database initialized")

#登入限制
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

#整理病歷資料
def organize_records(record):
    if record:

        # 時間處理：UTC -> UTC+8 (資料庫存的是UTC標準時間)
        if "created_at" in record and record["created_at"]:
            utc_time = datetime.datetime.strptime(record["created_at"], "%Y-%m-%d %H:%M:%S")
            local_time = utc_time + datetime.timedelta(hours=8)
            record["created_at"] = local_time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "user_id": record["username"],
        "sex": SEX_LABELS.get(record["sex"],record["sex"]),
        "age": record["age"],
        "narrow_pitch_range": YN_LABELS.get(record["narrow_pitch_range"],record["narrow_pitch_range"]),
        "decreased_volume": YN_LABELS.get(record["decreased_volume"],record["decreased_volume"]),
        "fatigue": YN_LABELS.get(record["fatigue"],record["fatigue"]),
        "dryness": YN_LABELS.get(record["dryness"],record["dryness"]),
        "lumping": YN_LABELS.get(record["lumping"],record["lumping"]),
        "heartburn": YN_LABELS.get(record["heartburn"],record["heartburn"]),
        "choking": YN_LABELS.get(record["choking"],record["choking"]),
        "eye_dryness": YN_LABELS.get(record["eye_dryness"],record["eye_dryness"]),
        "pnd": YN_LABELS.get(record["pnd"],record["pnd"]),
        "diabetes": YN_LABELS.get(record["diabetes"],record["diabetes"]),
        "hypertension": YN_LABELS.get(record["hypertension"],record["hypertension"]),
        "cad": YN_LABELS.get(record["cad"],record["cad"]),
        "head_and_neck_cancer": YN_LABELS.get(record["head_and_neck_cancer"],record["head_and_neck_cancer"]),
        "head_injury": YN_LABELS.get(record["head_injury"],record["head_injury"]),
        "cva": YN_LABELS.get(record["cva"],record["cva"]),
        "smoking": SMOKING_LABELS.get(record["smoking"],record["smoking"]),
        "ppd": record["ppd"],
        "drinking": DRINKING1_LABELS.get(record["drinking"],record["drinking"]),
        "frequency": DRINKING2_LABELS.get(record["frequency"],record["frequency"]),
        "onset_of_dysphonia": SYMPTOMS_LABELS.get(record["onset_of_dysphonia"],record["onset_of_dysphonia"]),
        "noise_at_work": WORKING_LABELS.get(record["noise_at_work"],record["noise_at_work"]),
        "diurnal_pattern": VOICE1_LABELS.get(record["diurnal_pattern"],record["diurnal_pattern"]),
        "occupational_vocal_demand": VOICE2_LABELS.get(record["occupational_vocal_demand"],record["occupational_vocal_demand"]),
        "vhi10": record["vhi10"],
        "created_at": record["created_at"]
    }

#病例分析模型
def predict_from_list(input_list, model_path, feature_names):
    # 建立 DataFrame
    df = pd.DataFrame([input_list], columns=feature_names)
    
    # 載入模型
    model = joblib.load(model_path)
    
    # 預測
    y_pred = model.predict(df)
    y_pred_proba = model.predict_proba(df)
    
    return y_pred, y_pred_proba

#嗓音模型
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
    # 計算 MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max_mfcc_length - mfcc.shape[1])), mode='constant')
    mfcc = mfcc[..., np.newaxis]
    X = np.array([mfcc])
    # 載入模型並預測
    model = load_model(model_path)
    y_pred = model.predict(X)
    y_pred_class = np.argmax(y_pred, axis=1)
    pred_label = class_names[y_pred_class[0]]
    pred_proba = y_pred[0]
    return pred_label, pred_proba

app = Flask(__name__) # 建立Application物件
app.secret_key = secrets.token_hex(16) # 安全字串
UPLOAD_FOLDER = 'uploads/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 網站首頁
@app.route('/')
def index():
    return render_template('index.html')

#註冊
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
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

#登入
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
            return redirect(url_for('Main'))
        else:
            return render_template("login.html",error="帳號或密碼錯誤")

    return render_template('login.html')

#登出
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/Main')# 主要功能頁面
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

@app.route('/upload_audio', methods=['POST'])# 上傳音訊檔案
def upload_audio():
    file = request.files['audio_file']
    if file and allowed_file(file.filename):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = file.filename.rsplit('.', 1)[1].lower()
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}.{ext}")
        file.save(save_path)
        return jsonify({"audio_path": save_path})
    else:
        return jsonify({"error": "請上傳 .wav 或 .mp3 或 .webm 音訊檔"}), 400
    
@app.route('/record_audio', methods=['POST'])  # 上傳錄音檔案
def record_audio():
    file = request.files['audio_data']
    if file:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}.webm")
        file.save(save_path)
        return jsonify({"audio_path": save_path})
    else:
        return jsonify({"error": "錄音失敗"}), 400

@app.route('/input_medical')# 手動輸入
@login_required # 限制訪問
def input_medical():
    return render_template('input_medical.html')

@app.route('/save_medical', methods=['POST'])# 存病歷資料
def save_medical():
    data = request.form

    # 計算 VHI-10 總分（10 個欄位相加）
    vhi10_total = sum(int(data.get(f'vhi{i}', 0)) for i in range(1, 11))

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO medical_records (
            user_id, sex, age, narrow_pitch_range, decreased_volume,
            fatigue, dryness, lumping, heartburn, choking, eye_dryness, pnd,
            diabetes, hypertension, cad, head_and_neck_cancer, head_injury, cva,
            smoking, ppd, drinking, frequency, onset_of_dysphonia, noise_at_work,
            diurnal_pattern, occupational_vocal_demand, vhi10
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session['user_id'],
        data['sex'],
        data['age'],
        data['narrow_pitch_range'],
        data['decreased_volume'],
        data['fatigue'],
        data['dryness'],
        data['lumping'],
        data['heartburn'],
        data['choking'],
        data['eye_dryness'],
        data['pnd'],
        data['diabetes'],
        data['hypertension'],
        data['cad'],
        data['head_and_neck_cancer'],
        data['head_injury'],
        data['cva'],
        data['smoking'],
        data['ppd'],
        data['drinking'],
        data['frequency'],
        data['onset_of_dysphonia'],
        data['noise_at_work'],
        data['diurnal_pattern'],
        data['occupational_vocal_demand'],
        vhi10_total
    ))
    conn.commit()
    conn.close()

    return redirect(url_for('Main'))

@app.route('/confirm_analysis', methods=['POST'])#確認後產生圖片顯示圖片
@login_required # 限制訪問
def confirm_analysis():
    audio_path = request.form.get("audio_path")

    # 撈最新病歷
    conn = get_db_connection()
    record = conn.execute("""
    SELECT medical_records.*, users.username
    FROM medical_records
    JOIN users ON medical_records.user_id = users.id
    WHERE username = ?
    ORDER BY medical_records.created_at DESC LIMIT 1""",(session['username'],)).fetchone()
    conn.close()

    record = dict(record) if record else None
    sample_list = [record[col] for col in MODEL_FEATURE_ORDER]
    record_id = record["id"]
    record_user = record["user_id"]
    record = organize_records(record)

    #病例分析模型
    model_path1 = 'best_svm_model.pkl'
    pred1, proba1= predict_from_list(sample_list, model_path1, feature_names)
    if pred1:
        if pred1==1:
            type1="結構性病變"
            conf1=proba1[0][0]
        if pred1==2:
            type1="非結構性病變"
            conf1=proba1[0][1]
        if pred1==3:
            type1="健康"
            conf1=proba1[0][2]

    #結構性病變嗓音模型
    type2=None
    conf2=None
    if pred1==1:
        model_path2 = 'smote_cnn_model_0.86.h5'
        class_names = [
        '1.Polyp & 2.Nodules', '13.Ulcer', '15.Dysplasia', '16.Laryngeal cancer', '17.Papilloma',
        '19.Scar', '20.Varix',
        '5.Vocal paresis', '6.Vocal palsy', "7.Reinke's edema", '8.Sulcus', '11.Vocal process granuloma', '12.Fibrous mass']
        chinese_name = ['聲帶息肉 & 聲帶結節','聲帶潰瘍','表皮異常增生','喉癌','乳突瘤',
                        '聲帶疤痕','微血管增生','單側聲帶輕癱','單側聲帶麻痹','慢性聲帶水腫'
                        ,'聲帶溝','聲帶突肉芽腫','纖維斑塊']
        pred2, proba2 = predict_audio_file(audio_path, model_path2, class_names)
        for i in range(len(class_names)):
            if pred2==class_names[i]:
                type2 = chinese_name[i]
                conf2 = proba2[i]
                break

    #非結構性病變嗓音模型
    if pred1==2:
        model_path2 = 'smote_non_cnn_model_0.84.h5'
        class_names = ['9.Muscle tension dysphonia','10.Presbyphonia','14.Spasmodic dysphonia','18.Tremor']

        pred2, proba2 = predict_audio_file(audio_path, model_path2, class_names)
        if pred2:
            if pred2=='9.Muscle tension dysphonia':
                type2="肌肉緊張性發聲異常"
                conf2=proba2[0]
            if pred2=='10.Presbyphonia':
                type2="嗓音老化"
                conf2=proba2[1]
            if pred2=='14.Spasmodic dysphonia':
                type2="聲帶痙攣"
                conf2=proba2[2]
            if pred2=='18.Tremor':
                type2="聲帶顫抖"
                conf2=proba2[3]
    type3=None
    #信心程度不高的警告
    if conf2 is not None:
        type3 = 1 if (conf1 <0.8 or conf2 < 0.8) else None
    else:
        type3 = 1 if (conf1 <0.8) else None

    #確認回傳
    if not audio_path or not os.path.exists(audio_path):
        return render_template("Main.html",
                               error="沒有找到音訊，請重新錄音或上傳",
                               record=record,
                               field_labels=FIELD_LABELS)

    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result = generate_audio_images(audio_path, timestamp)
        if result is None:
            return render_template("Main.html",
                                   error="音訊讀取錯誤，請確認檔案格式",
                                   record=record,
                                   field_labels=FIELD_LABELS)

        waveform_img, mel_img, mfcc_img = result

        #存結果
        with open(audio_path, "rb") as f:
            audio_blob = f.read()
        conn = get_db_connection()
        conn.execute("""
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
        conn.commit()
        conn.close()

        if pred1==3:
            return render_template("Main.html",
                               waveform_img=waveform_img,
                               mel_img=mel_img,
                               mfcc_img=mfcc_img,
                               record=record,
                               type1=type1,
                               conf1=conf1,
                               type2=type2,
                               conf2=conf2,
                               type3=type3,
                               field_labels=FIELD_LABELS)
        else:
            return render_template("Main.html",
                                waveform_img=waveform_img,
                                mel_img=mel_img,
                                mfcc_img=mfcc_img,
                                record=record,
                                type1=type1,
                                conf1=conf1,
                                type2=type2,
                                conf2=conf2,
                                type3=type3,
                                field_labels=FIELD_LABELS)
    except Exception as e:
        return render_template("Main.html",
                               error=f"分析失敗：{e}",
                               record=record,
                               field_labels=FIELD_LABELS)

if __name__ == '__main__':
    init_db()#啟動時確保資料表存在
    app.run(host='0.0.0.0',debug=True)