# app.py
import os
import uuid
import json
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, send_file
)
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash

from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use('Agg')  # без GUI
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# --- Инициализация приложения и БД ---

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me-in-production'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///similarity.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# папки для статики
STATIC_ROOT = os.path.join(app.root_path, 'static')
UPLOAD_FOLDER_REL = os.path.join('uploads')
RESULT_FOLDER_REL = os.path.join('results')

os.makedirs(os.path.join(STATIC_ROOT, UPLOAD_FOLDER_REL), exist_ok=True)
os.makedirs(os.path.join(STATIC_ROOT, RESULT_FOLDER_REL), exist_ok=True)

# --- Шрифты для кириллицы в PDF ---

FONT_REGULAR_NAME = 'DejaVuSans'
FONT_BOLD_NAME = 'DejaVuSans-Bold'

FONT_REGULAR_PATH = os.path.join(app.root_path, 'fonts', 'DejaVuSans.ttf')
FONT_BOLD_PATH = os.path.join(app.root_path, 'fonts', 'DejaVuSans-Bold.ttf')

USE_CUSTOM_FONTS = os.path.exists(FONT_REGULAR_PATH) and os.path.exists(FONT_BOLD_PATH)

if USE_CUSTOM_FONTS:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR_NAME, FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, FONT_BOLD_PATH))


# --- Модели ---

class User(db.Model):
    """Пользователь системы."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    analyses = db.relationship('Analysis', backref='user', lazy=True)


class Analysis(db.Model):
    """Результат одного анализа."""
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    image1_path = db.Column(db.String(255), nullable=False)
    image2_path = db.Column(db.String(255))
    method = db.Column(db.String(50))
    similarity = db.Column(db.Float)        # 0–100 (%)
    diff_path = db.Column(db.String(255))   # карта различий (для SSIM)
    mode = db.Column(db.String(20))         # 'pair' или 'search'
    extra = db.Column(db.Text)              # JSON

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class BaseImage(db.Model):
    """Изображения в базе для поиска похожих."""
    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feature_vector = db.Column(db.Text, nullable=False)  # JSON-список float


# --- Хелперы аутентификации ---

def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return User.query.get(user_id)


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            flash('Для этой операции необходимо войти в систему.')
            next_url = request.path
            return redirect(url_for('login', next=next_url))
        return view_func(*args, **kwargs)
    return wrapper


# --- Вспомогательные функции с изображениями ---

def save_uploaded_file(file_storage):
    """Сохранить загруженный файл и вернуть (rel_path, full_path)."""
    _, ext = os.path.splitext(file_storage.filename)
    ext = ext.lower() or '.png'
    filename = f"{uuid.uuid4().hex}{ext}"

    rel_path = os.path.join(UPLOAD_FOLDER_REL, filename)
    full_path = os.path.join(STATIC_ROOT, rel_path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    file_storage.save(full_path)
    return rel_path.replace("\\", "/"), full_path


def load_image(full_path, target_size=(256, 256)):
    img = Image.open(full_path).convert('RGB').resize(target_size)
    return np.array(img)


def compute_hist_feature(img_rgb):
    """
    Признак: 3D-гистограмма цвета (8x8x8).
    img_rgb – np.array (H,W,3) в RGB.
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hist = cv2.calcHist([bgr], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype('float32')


def compute_feature_similarity(f1, f2):
    """Косинусное сходство двух векторов признаков (0–1)."""
    f1 = f1.astype('float32')
    f2 = f2.astype('float32')
    f1_norm = np.linalg.norm(f1) + 1e-9
    f2_norm = np.linalg.norm(f2) + 1e-9
    sim = float(np.dot(f1, f2) / (f1_norm * f2_norm))
    return max(0.0, min(1.0, sim))


def compute_ssim(img1, img2):
    """SSIM (0..1) + карта различий 0..1."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    data_range = (gray1.max() - gray1.min()) or 255
    score, diff = ssim(gray1, gray2, full=True, data_range=data_range)

    diff_map = (1.0 - diff) / 2.0
    diff_map = np.clip(diff_map, 0.0, 1.0)
    return float(score), diff_map


def save_diff_figure(img1, img2, diff_map):
    """Сохранить картинку с двумя изображениями и картой различий."""
    filename = f"diff_{uuid.uuid4().hex}.png"
    rel_path = os.path.join(RESULT_FOLDER_REL, filename)
    full_path = os.path.join(STATIC_ROOT, rel_path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Изображение 1')

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Изображение 2')

    im = axes[2].imshow(diff_map, cmap='magma', vmin=0.0, vmax=1.0)
    axes[2].axis('off')
    axes[2].set_title('Карта различий')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(full_path, bbox_inches='tight')
    plt.close(fig)

    return rel_path.replace("\\", "/")


def ensure_base_image(rel_path):
    """Гарантировать, что картинка есть в BaseImage."""
    existing = BaseImage.query.filter_by(file_path=rel_path).first()
    if existing:
        return existing

    full_path = os.path.join(STATIC_ROOT, rel_path)
    if not os.path.exists(full_path):
        return None

    img = load_image(full_path)
    feat = compute_hist_feature(img)
    bi = BaseImage(
        file_path=rel_path,
        feature_vector=json.dumps(feat.tolist(), ensure_ascii=False)
    )
    db.session.add(bi)
    db.session.commit()
    return bi


def search_similar_images(query_img):
    """Поиск похожих изображений в базе по гистограммам."""
    q_feat = compute_hist_feature(query_img)
    results = []

    for bi in BaseImage.query.all():
        v = np.array(json.loads(bi.feature_vector), dtype='float32')
        sim = compute_feature_similarity(q_feat, v)
        results.append({
            'id': bi.id,
            'file_path': bi.file_path,
            'similarity': sim,
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results


def generate_pdf_report(analysis: Analysis):
    """Генерация PDF-отчёта по анализу."""
    reports_folder_rel = os.path.join(RESULT_FOLDER_REL, 'reports')
    reports_folder_full = os.path.join(STATIC_ROOT, reports_folder_rel)
    os.makedirs(reports_folder_full, exist_ok=True)

    pdf_filename = f"report_{analysis.id}.pdf"
    pdf_full_path = os.path.join(reports_folder_full, pdf_filename)

    c = canvas.Canvas(pdf_full_path, pagesize=A4)
    width, height = A4

    if USE_CUSTOM_FONTS:
        c.setFont(FONT_BOLD_NAME, 16)
    else:
        c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Отчёт по анализу сходства изображений")

    if USE_CUSTOM_FONTS:
        c.setFont(FONT_REGULAR_NAME, 12)
    else:
        c.setFont("Helvetica", 12)

    c.drawString(50, height - 90, f"ID анализа: {analysis.id}")
    c.drawString(50, height - 110,
                 f"Дата: {analysis.created_at.strftime('%Y-%m-%d %H:%M')}")
    c.drawString(50, height - 130, f"Режим: {analysis.mode}")
    c.drawString(50, height - 150, f"Метод: {analysis.method}")
    c.drawString(50, height - 170,
                 f"Коэффициент сходства: {analysis.similarity:.2f} %")

    img_rel = analysis.diff_path or analysis.image1_path
    if img_rel:
        img_full = os.path.join(STATIC_ROOT, img_rel)
        if os.path.exists(img_full):
            img_width = width - 100
            img_height = height / 2
            c.drawImage(
                img_full,
                50,
                height - 170 - img_height - 20,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True,
                mask='auto',
            )

    c.showPage()
    c.save()
    return pdf_full_path


# --- Маршруты приложения ---

@app.route('/', methods=['GET'])
def index():
    # ВАЖНО: теперь результат виден и гостю, если он его только что сделал
    user = get_current_user()
    last_result = session.get('last_result')
    return render_template('index.html', result=last_result, user=user)


@app.route('/analyze', methods=['POST'])
def analyze():
    mode = request.form.get('mode', 'pair')
    method = request.form.get('method', 'ssim')

    img1_file = request.files.get('image1')
    img2_file = request.files.get('image2')

    if mode == 'pair':
        if not img1_file or not img2_file or \
           img1_file.filename == '' or img2_file.filename == '':
            flash('Для сравнения двух изображений загрузите оба файла.')
            return redirect(url_for('index'))
    else:
        if not img1_file or img1_file.filename == '':
            flash('Для поиска похожих загрузите изображение.')
            return redirect(url_for('index'))

    rel1, full1 = save_uploaded_file(img1_file)
    rel2, full2 = None, None
    if mode == 'pair':
        rel2, full2 = save_uploaded_file(img2_file)

    img1 = load_image(full1)
    img2 = load_image(full2) if (mode == 'pair' and full2) else None

    result = {
        'mode': mode,
        'method': method,
        'method_label': 'SSIM (структурное сходство)'
        if method == 'ssim' else 'Гистограммный анализ',
        'image1': rel1,
        'image2': rel2,
        'similarity': None,
        'diff_image': None,
        'search_matches': []
    }

    ensure_base_image(rel1)
    if rel2:
        ensure_base_image(rel2)

    if mode == 'pair':
        if img2 is not None:
            if method == 'ssim':
                score, diff_map = compute_ssim(img1, img2)
                sim = max(0.0, min(1.0, score))
                diff_rel = save_diff_figure(img1, img2, diff_map)
                result['diff_image'] = diff_rel
            else:
                f1 = compute_hist_feature(img1)
                f2 = compute_hist_feature(img2)
                sim = compute_feature_similarity(f1, f2)
        else:
            flash('Не удалось загрузить второе изображение.')
            return redirect(url_for('index'))
    else:
        matches = search_similar_images(img1)
        result['search_matches'] = matches[:5]
        if matches:
            sim = matches[0]['similarity']
            result['image2'] = matches[0]['file_path']
        else:
            sim = 0.0

    result['similarity'] = round(sim * 100.0, 2)

    session['last_result'] = result
    flash('Анализ выполнен. Нажмите «Сохранить результат», чтобы добавить в историю.')
    return redirect(url_for('index'))


@app.route('/save', methods=['POST'])
@login_required
def save_result():
    data = session.get('last_result')
    if not data:
        flash('Нет результата анализа для сохранения.')
        return redirect(url_for('index'))

    user = get_current_user()

    analysis = Analysis(
        image1_path=data['image1'],
        image2_path=data.get('image2'),
        method=data['method'],
        similarity=data['similarity'],
        diff_path=data.get('diff_image'),
        mode=data['mode'],
        extra=json.dumps(data.get('search_matches', []), ensure_ascii=False),
        user_id=user.id if user else None
    )
    db.session.add(analysis)

    ensure_base_image(data['image1'])
    if data.get('image2'):
        ensure_base_image(data['image2'])

    db.session.commit()

    flash('Результат сохранён в историю.')
    return redirect(url_for('history'))


@app.route('/history', methods=['GET'])
@login_required
def history():
    user = get_current_user()
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('history.html', analyses=analyses, user=user)


@app.route('/history/delete/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    user = get_current_user()
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != user.id:
        flash('Нельзя удалить чужой анализ.')
        return redirect(url_for('history'))

    db.session.delete(analysis)
    db.session.commit()
    flash('Запись успешно удалена.')
    return redirect(url_for('history'))


@app.route('/report/<int:analysis_id>/pdf', methods=['GET'])
@login_required
def report_pdf(analysis_id):
    user = get_current_user()
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != user.id:
        flash('Нельзя сформировать отчёт по чужому анализу.')
        return redirect(url_for('history'))

    pdf_path = generate_pdf_report(analysis)
    return send_file(pdf_path, as_attachment=True)


# --- Аутентификация ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if get_current_user():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        password2 = request.form.get('password2') or ''

        if not username or not password:
            flash('Введите логин и пароль.')
            return redirect(url_for('register'))

        if password != password2:
            flash('Пароли не совпадают.')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким логином уже существует.')
            return redirect(url_for('register'))

        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash('Регистрация успешно завершена. Теперь войдите в систему.')
        return redirect(url_for('login'))

    return render_template('register.html', user=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if get_current_user():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        next_url = request.args.get('next') or url_for('index')

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Неверный логин или пароль.')
            return redirect(url_for('login', next=request.args.get('next')))

        session['user_id'] = user.id
        session.pop('last_result', None)

        flash('Вы успешно вошли в систему.')
        return redirect(next_url)

    next_url = request.args.get('next')
    return render_template('login.html', next=next_url, user=None)


@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    session.pop('last_result', None)
    flash('Вы вышли из системы.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # Для облачного хостинга используем 0.0.0.0
    # PORT берем из переменной окружения или используем 5000 по умолчанию
    port = int(os.environ.get('PORT', 5000))
    
    # Отключаем debug режим для продакшена
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
