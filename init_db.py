import sqlite3#測試建立資料庫

def init_db():
    conn = sqlite3.connect("database.db")  # 建立 database.db
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
        ppd REAL,
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
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_db()