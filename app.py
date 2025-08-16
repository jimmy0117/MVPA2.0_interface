from flask import Flask, render_template, request, redirect, url_for, jsonify
from pathlib import Path
from pydub import AudioSegment
import datetime
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os
import sqlite3
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
    "created_at": "建立時間"
}

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
        result TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (record_id) REFERENCES medical_records(id)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized")

app = Flask(__name__)# 建立Application物件
UPLOAD_FOLDER = 'uploads/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')# 網站首頁(打算用登入但還沒用)
def index():
    return render_template('index.html')

@app.route('/Main')# 主要功能頁面
def Main():
    # 讀最新病歷
    conn = get_db_connection()
    record = conn.execute("SELECT * FROM medical_records ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()

    record = dict(record) if record else None
    if record and "id" in record:
        del record["id"]

    return render_template("Main.html", record=record,field_labels=FIELD_LABELS)

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
        1,  # 先暫時寫死 user_id=1
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
def confirm_analysis():
    audio_path = request.form.get("audio_path")

    # 撈最新病歷
    conn = get_db_connection()
    record = conn.execute("SELECT * FROM medical_records ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    record = dict(record) if record else None
    if record and "id" in record:
        del record["id"]

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

        return render_template("Main.html",
                               waveform_img=waveform_img,
                               mel_img=mel_img,
                               mfcc_img=mfcc_img,
                               record=record,
                               field_labels=FIELD_LABELS)
    except Exception as e:
        return render_template("Main.html",
                               error=f"分析失敗：{e}",
                               record=record,
                               field_labels=FIELD_LABELS)

if __name__ == '__main__':
    init_db()#啟動時確保資料表存在
    app.run(host='0.0.0.0',debug=True, use_reloader=False)