import sqlite3  # 測試建立資料庫

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # 使用者、醫生表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    """)

    # 病歷表（含 VHI 題目）
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
        vhi1 INTEGER, vhi2 INTEGER, vhi3 INTEGER, vhi4 INTEGER, vhi5 INTEGER,
        vhi6 INTEGER, vhi7 INTEGER, vhi8 INTEGER, vhi9 INTEGER, vhi10_q INTEGER,
        vhi10 INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 自動遷移：補欄位
    cursor.execute("PRAGMA table_info(medical_records)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col in ["vhi1","vhi2","vhi3","vhi4","vhi5","vhi6","vhi7","vhi8","vhi9","vhi10_q"]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE medical_records ADD COLUMN {col} INTEGER DEFAULT 0")

    # 分析結果表
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
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_db()