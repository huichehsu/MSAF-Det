# Model

A custom object detection model built on the [Ultralytics](https://github.com/ultralytics/ultralytics) framework, supporting **VisDrone** and **UAVDT** drone-view datasets.

---

## Requirements

```bash
pip install ultralytics
```

---

## Dataset Preparation

Download and place the datasets as follows:

| Dataset | Folder Structure |
|---------|-----------------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | `visdrone/VisDrone2019-DET-train/` `visdrone/VisDrone2019-DET-val/` |
| [UAVDT](https://sites.google.com/view/grli-uavdt/) | `UAVDT/UAVDT-train/` `UAVDT/UAVDT-test/` |

Set the `path:` field in `visdrone.yaml` or `UAVDT.yaml` to your dataset root directory if needed.

---

## Training

```bash
python train.py
```

To switch datasets, change the `data` argument in `train.py`:

| Dataset | `data=` |
|---------|---------|
| VisDrone | `'visdrone.yaml'` |
| UAVDT | `'UAVDT.yaml'` |

---

## Validation

```bash
python validation.py
```

To switch datasets, change the `data` argument and the `weight` path in `validation.py`:

| Dataset | `data=` |
|---------|---------|
| VisDrone | `'visdrone.yaml'` |
| UAVDT | `'UAVDT.yaml'` |

Set `weight` to your trained checkpoint, e.g., `./train/<run_name>/weights/best.pt`.

---

## Citation

If you use this work, please cite according to [CITATION.cff](CITATION.cff).
