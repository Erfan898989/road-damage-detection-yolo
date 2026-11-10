# Road Damage Detection with YOLO11n
# تشخیص خرابی‌های جاده با YOLO11n

A computer vision project that detects three types of road damage using **YOLO11n**:
- **Pothole** — چاله
- **Cracks** — ترک آسفالت
- **Open Manhole** — دریچه باز

این پروژه با هدف شناسایی سه نوع خرابی جاده با استفاده از مدل YOLO11n ساخته شده است.

## Project overview | معرفی پروژه

The dataset was prepared in YOLO format and split into training, validation, and test sets. The model was trained in Google Colab using an NVIDIA Tesla T4 GPU.

داده‌ها به قالب YOLO تبدیل و به سه بخش آموزش، اعتبارسنجی و آزمون تقسیم شدند. آموزش مدل در Google Colab و با GPU مدل Tesla T4 انجام شد.

| Item | Value |
|---|---:|
| Total images | 718 |
| Training images | 502 |
| Validation images | 143 |
| Test images | 73 |
| Training epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Random seed | 42 |
| Model | YOLO11n |

## Dataset split | تقسیم داده‌ها

| Split | Images | Bounding boxes | Background images |
|---|---:|---:|---:|
| Train | 502 | 571 | 65 |
| Validation | 143 | 168 | 22 |
| Test | 73 | 87 | 13 |
| **Total** | **718** | **826** | **100** |

### Classes | کلاس‌ها

| Class ID | Class name | نام فارسی |
|---:|---|---|
| 0 | pothole | چاله |
| 1 | cracks | ترک آسفالت |
| 2 | open_manhole | دریچه باز |

## Training | آموزش

The model was trained for 100 epochs with the following main settings:

- Model: `yolo11n.pt`
- Epochs: 100
- Image size: 640
- Batch size: 16
- Patience: 20
- Device: CUDA (Tesla T4)
- Seed: 42

The best checkpoint was saved as `best.pt` (approximately 5.26 MB).

## Evaluation results | نتایج ارزیابی

### Validation set

| Metric | Score |
|---|---:|
| Precision | 83.3% |
| Recall | 67.9% |
| mAP50 | 76.5% |
| mAP50-95 | 40.4% |

### Test set

| Metric | Score |
|---|---:|
| Precision | 82.4% |
| Recall | 79.2% |
| mAP50 | 82.2% |
| mAP50-95 | 45.0% |

### Test results by class | نتایج هر کلاس در مجموعه آزمون

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pothole | 81.4% | 68.9% | 79.9% | 39.8% |
| Cracks | 77.4% | 68.8% | 71.9% | 26.5% |
| Open Manhole | 88.3% | 100.0% | 95.0% | 68.8% |

**Important note:** The test set contains only 87 labeled objects: 61 potholes, 16 cracks, and 10 open manholes. The per-class results—especially for open manholes—should therefore be interpreted cautiously and do not guarantee the same performance on all real-world images.

**نکته مهم:** مجموعه آزمون فقط شامل ۸۷ شیء برچسب‌گذاری‌شده است؛ بنابراین به‌خصوص نتیجه کلاس دریچه باز که فقط ۱۰ نمونه دارد، برای تعمیم به همه تصاویر دنیای واقعی کافی نیست.

## Inference | اجرای مدل

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run prediction on an image:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    source="path/to/road_image.jpg",
    conf=0.25,
    imgsz=640,
    save=True
)
```

The prediction image is saved by Ultralytics under a `runs/detect/predict/` directory.

## Repository structure | ساختار مخزن

```text
road-damage-detection-yolo/
├── README.md
├── requirements.txt
├── .gitignore
├── data.yaml
├── best.pt                 # trained weights (to be added)
├── notebooks/              # training and evaluation notebook (to be added)
└── examples/               # prediction examples (to be added)
```

## Dataset and license | دیتاست و مجوز

The dataset was assembled from a supplied dataset archive and reorganized for this project. The original dataset source URL and license have not yet been verified. Please check the original dataset's terms before redistributing the dataset or its images.

منبع اینترنتی و مجوز رسمی دیتاست هنوز تأیید نشده است. پیش از انتشار یا بازتوزیع تصاویر دیتاست، شرایط استفاده و مجوز منبع اصلی باید بررسی شود.

## Limitations and future work | محدودیت‌ها و کارهای آینده

- Evaluate on a larger and more diverse test set.
- Add more examples for cracks and open manholes.
- Review false positives and missed detections.
- Add prediction examples and the training notebook to this repository.

## Acknowledgements

- Ultralytics YOLO
- Google Colab

---
