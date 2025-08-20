from flask import Flask, request, render_template_string, url_for, redirect
import os, uuid
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# -------------------------
# إعداد مجلدات
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
DEBUG_FOLDER = os.path.join(BASE_DIR, "static", "debug")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# -------------------------
# دوال Dice
# -------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# -------------------------
# تحميل الموديلات
# -------------------------
DETECTION_MODEL_PATH = os.path.join("models", "brain_tumor_resnet50.keras")
SEGMENTATION_MODEL_PATH = os.path.join("models", "brain_tumor_unet.keras")

print("Loading models...")
detection_model = keras.models.load_model(DETECTION_MODEL_PATH, compile=False)
segmentation_model = keras.models.load_model(
    SEGMENTATION_MODEL_PATH,
    custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef},
    compile=False
)
print("Models loaded.")

# -------------------------
# Utility functions
# -------------------------
last_uploaded_file = None  # لتخزين آخر صورة مرفوعة مؤقتاً

def extract_hwcs(inp_shape, default=(224,224,3)):
    if inp_shape is None: return default
    if len(inp_shape) == 4:
        _, h, w, c = inp_shape
        if h is None or w is None or c is None:
            return default
        return (int(h), int(w), int(c))
    return default

det_h, det_w, det_c = extract_hwcs(detection_model.input_shape, default=(224,224,3))
seg_h, seg_w, seg_c = extract_hwcs(segmentation_model.input_shape, default=(256,256,1))

def prepare_for_detection(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
    arr = np.expand_dims(img.astype(np.float32), axis=0)
    try:
        arr = resnet_preprocess(arr)
    except Exception:
        arr = arr / 255.0
    return arr

def prepare_for_segmentation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (seg_w, seg_h), interpolation=cv2.INTER_LINEAR)
    img_float = img.astype(np.float32)/255.0
    dbg_name = f"pre_{uuid.uuid4().hex}.png"
    dbg_path = os.path.join(DEBUG_FOLDER, dbg_name)
    cv2.imwrite(dbg_path, (img_float*255).astype(np.uint8))
    if seg_c==3: img_float = np.stack([img_float]*3, axis=-1)
    else: img_float = np.expand_dims(img_float, axis=-1)
    arr = np.expand_dims(img_float, axis=0)
    return arr, dbg_path

def postprocess_and_save_mask_overlay(orig_path, pred_mask, mask_out_path, overlay_out_path, tumor_only_out_path):
    orig = cv2.imread(orig_path)
    orig_h, orig_w = orig.shape[:2]
    pred = np.array(pred_mask)
    if pred.max()>1.0: pred=1.0/(1.0+np.exp(-pred))
    if pred.ndim==3 and pred.shape[-1]==1: pred=np.squeeze(pred,axis=-1)
    elif pred.ndim==3 and pred.shape[-1]>1: pred=pred[...,-1]
    mask_bin=(pred>0.5).astype(np.uint8)*255
    mask_resized=cv2.resize(mask_bin,(orig_w,orig_h),interpolation=cv2.INTER_NEAREST)
    min_area=max(50,int(0.0005*orig_h*orig_w))
    contours,_=cv2.findContours(mask_resized.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    clean_mask=np.zeros_like(mask_resized)
    for cnt in contours:
        if cv2.contourArea(cnt)>=min_area:
            cv2.drawContours(clean_mask,[cnt],-1,255,thickness=cv2.FILLED)
    if np.count_nonzero(clean_mask)==0:
        cv2.imwrite(mask_out_path, clean_mask)
        black=np.zeros_like(orig)
        cv2.imwrite(overlay_out_path, black)
        cv2.imwrite(tumor_only_out_path, black)
        return False
    else:
        cv2.imwrite(mask_out_path, clean_mask)
        colored=np.zeros_like(orig)
        colored[clean_mask>127]=(0,0,255)
        overlay=cv2.addWeighted(orig,0.7,colored,0.3,0)
        cv2.imwrite(overlay_out_path, overlay)
        tumor_only=np.zeros_like(orig)
        tumor_only[clean_mask>127]=orig[clean_mask>127]
        cv2.imwrite(tumor_only_out_path, tumor_only)
        return True

# -------------------------
# HTML Template مع loader كامل الشاشة وزر segmentation
# -------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Brain Tumor Detection & Segmentation</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{background:linear-gradient(to right,#1f1c2c,#928dab);font-family:'Segoe UI',sans-serif;color:#fff;min-height:100vh;margin:0;}
.card{border-radius:20px;box-shadow:0 12px 40px rgba(0,0,0,0.35);padding:30px;margin-bottom:30px;background:rgba(255,255,255,0.95);transition:transform 0.3s ease,box-shadow 0.3s ease;}
.card:hover{transform:translateY(-7px);box-shadow:0 25px 50px rgba(0,0,0,0.4);}
h3,h4,h6{margin-bottom:0.5rem;color:#333;}
img{width:100%;max-height:350px;object-fit:contain;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.2);margin-bottom:10px;transition: transform 0.3s ease;}
img:hover{transform: scale(1.03);}
.btn-primary{background:linear-gradient(to right,#ff416c,#ff4b2b);border:none;font-weight:bold;transition:all 0.3s ease;}
.btn-primary:hover{background:linear-gradient(to right,#ff4b2b,#ff416c);transform:scale(1.05);}
input[type="file"]{cursor:pointer;}
.preview-img{display:none;max-width:100%;margin-top:15px;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.2);}
.row-imgs{display:flex;flex-wrap:wrap;gap:20px;justify-content:center;}
.img-container{flex:1 1 45%;text-align:center;}
/* Loader overlay */
#loader-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:9999;align-items:center;justify-content:center;}
.loader-circle{border:8px solid #f3f3f3;border-top:8px solid #3498db;border-radius:50%;width:70px;height:70px;animation:spin 1s linear infinite;}
@keyframes spin{100%{transform:rotate(360deg);}}
</style>
<script>
function showLoader(){document.getElementById('loader-overlay').style.display='flex';}
function previewImage(event){
    const input=event.target;
    const preview=document.getElementById('preview-img');
    if(input.files && input.files[0]){
        const reader=new FileReader();
        reader.onload=function(e){preview.src=e.target.result;preview.style.display='block';}
        reader.readAsDataURL(input.files[0]);
    }
}
function runSegmentation(){ 
    document.getElementById('action-input').value='segmentation';
    showLoader();
    document.getElementById('main-form').submit();
}
</script>
</head>
<body>
<div id="loader-overlay"><div class="loader-circle"></div></div>
<div class="container py-5">
<div class="card">
<h3 class="text-center mb-4">🧠 Brain Tumor — Detection & Segmentation</h3>
<form method="post" enctype="multipart/form-data" id="main-form" onsubmit="showLoader()">
<div class="row g-3 align-items-center">
<div class="col-md-7">
<input class="form-control" type="file" name="image" accept="image/*" {% if not uploaded_url %} required onchange="previewImage(event)" {% endif %}>
<img id="preview-img" class="preview-img" src="{{ uploaded_url or '' }}">
</div>
<div class="col-md-3">
<select class="form-select" name="action" id="action-input" required>
<option value="detection">Detection</option>
<option value="segmentation">Segmentation</option>
</select>
</div>
<div class="col-md-2">
<button class="btn btn-primary w-100 py-2">Run</button>
</div>
</div>
</form>

{% if uploaded_url %}
<hr>
<h6 class="mb-3">Results</h6>
<div class="row row-imgs">
<div class="img-container">
<h6>Uploaded Image</h6>
<img src="{{ uploaded_url }}" alt="uploaded">
</div>

{% if action=='detection' and result_text %}
<div class="img-container">
<h6>Detection</h6>
<div class="p-3 bg-white rounded shadow-sm"><h4>{{ result_text }}</h4></div>
</div>
<div class="col-12 mt-2 text-center">
<button class="btn btn-primary" onclick="runSegmentation()">Run Segmentation on same image</button>
</div>
{% elif action=='segmentation' %}
<div class="img-container">
<h6>Mask</h6>
{% if mask_url %}<img src="{{ mask_url }}">{% endif %}
</div>
<div class="img-container">
<h6>Overlay (red = tumor)</h6>
{% if overlay_url %}<img src="{{ overlay_url }}">{% endif %}
</div>
<div class="img-container">
<h6>Tumor Only</h6>
{% if tumor_only_url %}<img src="{{ tumor_only_url }}">{% endif %}
</div>
<p class="small-note mt-2">Debug images in <code>/static/debug/</code></p>
{% endif %}
</div>
{% endif %}
</div>
</div>
</body>
</html>
"""

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    global last_uploaded_file
    uploaded_url = mask_url = overlay_url = tumor_only_url = result_text = action = None

    if request.method=="POST":
        action = request.form.get("action", "detection")
        # إذا كانت الصورة جديدة
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(save_path)
            uploaded_url = url_for('static', filename=f"uploads/{unique_name}")
            last_uploaded_file = save_path
        elif last_uploaded_file:
            save_path = last_uploaded_file
            uploaded_url = url_for('static', filename=f"uploads/{os.path.basename(save_path)}")
        else:
            return redirect("/")

        if action=="detection":
            x=prepare_for_detection(save_path)
            preds=detection_model.predict(x).reshape(-1)
            if preds.size==1:
                tumor_prob=float(preds[0]); no_tumor_prob=1.0-tumor_prob
                label,prob=("Tumor",tumor_prob) if tumor_prob>=no_tumor_prob else ("No Tumor",no_tumor_prob)
            else:
                idx=int(np.argmax(preds)); prob=float(preds[idx]); label="Tumor" if idx==1 else "No Tumor"
            result_text=f"{label} — {prob*100:.2f}%"

        elif action=="segmentation":
            x,dbg_pre_path=prepare_for_segmentation(save_path)
            pred=segmentation_model.predict(x)[0]
            mask_fname=f"mask_{os.path.basename(save_path)}"; overlay_fname=f"overlay_{os.path.basename(save_path)}"; tumor_only_fname=f"tumoronly_{os.path.basename(save_path)}"
            mask_path=os.path.join(RESULT_FOLDER, mask_fname)
            overlay_path=os.path.join(RESULT_FOLDER, overlay_fname)
            tumor_only_path=os.path.join(RESULT_FOLDER, tumor_only_fname)
            has_tumor=postprocess_and_save_mask_overlay(save_path,pred,mask_path,overlay_path,tumor_only_path)
            mask_url=url_for('static', filename=f"results/{mask_fname}")
            overlay_url=url_for('static', filename=f"results/{overlay_fname}")
            tumor_only_url=url_for('static', filename=f"results/{tumor_only_fname}")
            result_text="Tumor detected" if has_tumor else "No tumor detected"

    return render_template_string(INDEX_HTML,
                                  uploaded_url=uploaded_url,
                                  mask_url=mask_url,
                                  overlay_url=overlay_url,
                                  tumor_only_url=tumor_only_url,
                                  result_text=result_text,
                                  action=action)

if __name__=="__main__":
    app.run(debug=True)
