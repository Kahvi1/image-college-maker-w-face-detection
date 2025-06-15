import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import math
import tempfile
import shutil
from io import BytesIO
from pathlib import Path

# Initialize the face detection class
class FaceDetectionCollage:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return largest_face

    def crop_face(self, img, face_coords, padding_ratio=0.2):
        if face_coords is None:
            return None

        x, y, w, h = face_coords
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img.shape[1], x + w + padding_x)
        y2 = min(img.shape[0], y + h + padding_y)

        cropped_face = img[y1:y2, x1:x2]
        return cropped_face

    def resize_image(self, img, target_size=(200, 200)):
        return cv2.resize(img, target_size)

    def create_collage(self, cropped_faces, collage_size=(400, 400)):
        if not cropped_faces:
            return None

        num_faces = len(cropped_faces)
        cols = math.ceil(math.sqrt(num_faces))
        rows = math.ceil(num_faces / cols)
        face_width = collage_size[0] // cols
        face_height = collage_size[1] // rows

        collage = np.zeros((collage_size[1], collage_size[0], 3), dtype=np.uint8)
        collage.fill(255)  # White background

        for i, face in enumerate(cropped_faces):
            row = i // cols
            col = i % cols
            resized_face = cv2.resize(face, (face_width, face_height))
            y_start = row * face_height
            y_end = y_start + face_height
            x_start = col * face_width
            x_end = x_start + face_width

            collage[y_start:y_end, x_start:x_end] = resized_face

        return collage

    def process_images(self, uploaded_files):
        cropped_faces = []
        for img_file in uploaded_files:
            img = Image.open(img_file)
            img = np.array(img)
            face_coords = self.detect_face(img)
            cropped_face = self.crop_face(img, face_coords)
            if cropped_face is not None:
                resized_face = self.resize_image(cropped_face)
                cropped_faces.append(resized_face)

        return cropped_faces

# Streamlit UI for face detection and collage creation
def main():
    st.title("Face Detection and Collage System")

    st.write(
        "Upload your images (JPG, PNG, etc.) and we'll detect faces, crop them, and create a collage!"
    )

    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        face_detector = FaceDetectionCollage()

        st.write(f"Processing {len(uploaded_files)} images...")
        cropped_faces = face_detector.process_images(uploaded_files)

        if cropped_faces:
            st.write("### Cropped Faces")
            for i, cropped_face in enumerate(cropped_faces):
                st.image(cropped_face, caption=f"Face {i+1}", use_column_width=True)

            st.write("### Creating Collage...")
            collage = face_detector.create_collage(cropped_faces, collage_size=(600, 600))
            if collage is not None:
                st.image(collage, caption="Collage", use_column_width=True)

                # Option to download collage
                collage_image = Image.fromarray(collage)
                img_bytes = BytesIO()
                collage_image.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                st.download_button(
                    label="Download Collage",
                    data=img_bytes,
                    file_name="face_collage.png",
                    mime="image/png"
                )

                # Option to download individual faces
                for i, cropped_face in enumerate(cropped_faces):
                    img = Image.fromarray(cropped_face)
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    st.download_button(
                        label=f"Download Face {i+1}",
                        data=img_bytes,
                        file_name=f"face_{i+1}.png",
                        mime="image/png"
                    )
        else:
            st.write("No faces detected in the uploaded images.")

if __name__ == "__main__":
    main()
