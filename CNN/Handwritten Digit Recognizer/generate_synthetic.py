import os
import cv2
import numpy as np

output_dir = "data/custom"
os.makedirs(output_dir, exist_ok=True)

fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX
]

for digit in range(10):
    folder = os.path.join(output_dir, str(digit))
    os.makedirs(folder, exist_ok=True)

    for i in range(300):  # 300 per digit → 3000 total
        img = np.ones((100, 100), dtype=np.uint8) * 255

        font = np.random.choice(fonts)
        scale = np.random.uniform(1.5, 2.5)
        thickness = np.random.randint(2, 6)

        x = np.random.randint(5, 25)
        y = np.random.randint(60, 90)

        cv2.putText(
            img,
            str(digit),
            (x, y),
            font,
            scale,
            (0,),
            thickness,
            cv2.LINE_AA
        )

        # 🔥 Add rotation
        angle = np.random.uniform(-25, 25)
        M = cv2.getRotationMatrix2D((50, 50), angle, 1)
        img = cv2.warpAffine(img, M, (100, 100))

        # 🔥 Add noise
        noise = np.random.randint(0, 40, (100, 100), dtype=np.uint8)
        img = cv2.subtract(img, noise)

        # 🔥 Slight blur
        if np.random.rand() > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        img = cv2.resize(img, (28, 28))

        cv2.imwrite(os.path.join(folder, f"{i}.png"), img)

print("✅ Synthetic dataset created!")