# ✍️ ระบบทำนายและจัดเก็บข้อมูลตัวเลขไทยจากลายมือ (๒๖-๓๐)

โปรเจกต์นี้เป็น Web Application แบบ Full-stack สำหรับ **การจัดเก็บข้อมูล (Data Collection)** และ **ทำนายลายมือตัวเลขไทย (๒๖-๓๐)** ด้วย Machine Learning (CNN) โดยออกแบบมาให้ใช้งานง่าย รวดเร็ว และรองรับการทำงานทั้งฝั่งผู้ใช้งานทั่วไป (User) และผู้ดูแลระบบ (Admin)

## 🌐 ลิงก์ใช้งานจริง (Live Demo)
- **เว็บทำนายลายมือ (Prediction App):** [https://project322-number26-30.onrender.com/](https://project322-number26-30.onrender.com/)
- **เว็บเก็บข้อมูล (Data Collection Tool):** [https://miyomui.pythonanywhere.com/](https://miyomui.pythonanywhere.com/)

---

## ✨ ฟีเจอร์หลัก (Features)

### 1. ระบบจัดการโมเดลหลังบ้าน (Admin Dashboard)
- **Hot-Reloading:** รองรับการอัปโหลดไฟล์โมเดล (`.keras`, `.h5`) และสลับการใช้งานโมเดลได้ทันทีโดยไม่ต้อง Restart Server
- **Real-time Evaluation:** มีระบบประเมินผลโมเดล (Accuracy, Precision, Recall, F1-Score) แบบเรียลไทม์ โดยใช้ Dataset ของจริงในโฟลเดอร์
- **Error Analysis:** ระบบแสดงผลคลาสที่โมเดลสับสนมากที่สุดจาก Confusion Matrix
- **In-Memory Caching:** ประมวลผลรูปภาพ 2,500+ รูป ได้ภายในไม่ถึง 1 วินาที!

### 2. ระบบทำนายตัวเลข (User Page)
- **Interactive Canvas:** กระดานวาดรูปบนเว็บที่รองรับทั้งเมาส์และการสัมผัสบนมือถือ/แท็บเล็ต
- **Auto Pre-processing:** ระบบจะทำการแปลงรูปที่วาดให้อัตโนมัติ (แปลงเป็นพื้นดำ ตัวอักษรสีขาว ขนาด 64x64) เพื่อส่งเข้าโมเดล
- **Instant Prediction:** คืนผลลัพธ์การทำนายทันที พร้อมแสดงความน่าจะเป็น (Confidence Score) ของตัวเลขอื่นๆ

### 3. ระบบจัดเก็บชุดข้อมูล (Data Collection Tool)
- **Collect & Manage:** ระบบสำหรับเก็บภาพตัวเลขไทยจากการเขียน แบ่งหมวดหมู่ (๒๖-๓๐) ลงโฟลเดอร์ให้อัตโนมัติ
- **Gallery Management:** มีระบบดูรูปล่าสุด และสามารถกดลบรูปที่เขียนผิดพลาดได้ทันที

---

## 🛠️ โครงสร้างของโปรเจกต์ (Project Structure)
```text
project_aie322/
│
├── backend/
│   └── app.py                # เซิร์ฟเวอร์หลักสำหรับรัน Model และ API (Flask)
│
├── data_collection/
│   ├── app.py                # เซิร์ฟเวอร์ย่อยสำหรับเก็บข้อมูล
│   └── templates/            # หน้าเว็บระบบเก็บข้อมูล
│
├── frontend/
│   ├── user.html             # หน้าเว็บสำหรับผู้ใช้งาน (วาดรูปทำนาย)
│   └── admin.html            # หน้าเว็บ Dashboard สำหรับแอดมิน
│
├── dataset/                  # โฟลเดอร์เก็บรูปภาพสำหรับทดสอบและเทรน (26, 27, 28, 29, 30)
├── model/                    # โฟลเดอร์เก็บไฟล์โมเดล
└── requirements.txt          # ไฟล์รายชื่อไลบรารีที่จำเป็นสำหรับ Deploy
```

---

## 🚀 การติดตั้งและรันในเครื่อง (Local Setup)

1. **ติดตั้ง Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **รันเซิร์ฟเวอร์หลัก (หน้าทำนายและแอดมิน):**
   ```bash
   python backend/app.py
   ```
   เข้าไปที่:
   - User Page: `http://127.0.0.1:5000/`
   - Admin Page: `http://127.0.0.1:5000/admin`

3. **รันระบบเก็บข้อมูล (Data Collection):**
   ```bash
   cd data_collection
   python app.py
   ```
   เข้าไปที่ `http://127.0.0.1:5000/` (พอร์ตอาจจะเปลี่ยนเป็น 5001 หากเซิร์ฟเวอร์หลักรันอยู่)




