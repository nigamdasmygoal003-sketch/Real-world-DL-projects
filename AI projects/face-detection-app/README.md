## YOLO + CNN mask detection

This project uses a 2-stage pipeline:

1. YOLO detects the target region.
2. The detected crop is passed to a lightweight CNN classifier.
3. The CNN predicts `Mask` or `No Mask`.

## Train the CNN

Create your dataset in ImageFolder format:

```text
data/
|-- mask_dataset/
|   |-- Mask/
|   |   |-- image1.jpg
|   |   `-- ...
|   `-- No Mask/
|       |-- image2.jpg
|       `-- ...
```

Run training from the project root:

```powershell
python train_mask_model.py --data-dir data/mask_dataset --epochs 10
```

This will save the trained weights to `models/mask_model.pth`.

## Automatic dataset download

You can automatically download a public Hugging Face face-mask dataset repo and convert its YOLO annotations into cropped classification images:

```text
data/
|-- mask_dataset/
|   |-- Mask/
|   `-- NoMask/
```

Install dependencies and run:

```powershell
pip install -r requirements.txt
python download_mask_dataset.py
```

Then train:

```powershell
python train_mask_model.py --data-dir data/mask_dataset --epochs 10
```

If the dataset class ids are reversed in your local download, rerun with:

```powershell
python download_mask_dataset.py --mask-class-id 1 --nomask-class-id 0
```

## Run the Streamlit app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

If `models/mask_model.pth` is present, the app will run full inference:

`Webcam -> YOLO detect -> crop -> CNN classify -> display result`
