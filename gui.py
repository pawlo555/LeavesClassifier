from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class_map = {0: "Apple___Apple_scab",
             1: "Apple___Black_rot",
             2: "Apple___Cedar_apple_rust",
             3: "Apple___healthy",
             4: "Blueberry___healthy",
             5: "Cherry_(including_sour)___Powdery_mildew",
             6: "Cherry_(including_sour)___healthy",
             7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
             8: "Corn_(maize)___Common_rust_",
             9: "Corn_(maize)___Northern_Leaf_Blight",
             10: "Corn_(maize)___healthy",
             11: "Grape___Black_rot",
             12: "Grape___Esca_(Black_Measles)",
             13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
             14: "Grape___healthy",
             15: "Orange___Haunglongbing_(Citrus_greening)",
             16: "Peach___Bacterial_spot",
             17: "Peach___healthy",
             18: "Pepper,_bell___Bacterial_spot",
             19: "Pepper,_bell___healthy",
             20: "Potato___Early_blight",
             21: "Potato___Late_blight",
             22: "Potato___healthy",
             23: "Raspberry___healthy",
             24: "Soybean___healthy",
             25: "Squash___Powdery_mildew",
             26: "Strawberry___Leaf_scorch",
             27: "Strawberry___healthy",
             28: "Tomato___Bacterial_spot",
             29: "Tomato___Early_blight",
             30: "Tomato___Late_blight",
             31: "Tomato___Leaf_Mold",
             32: "Tomato___Septoria_leaf_spot",
             33: "Tomato___Spider_mites Two-spotted_spider_mite",
             34: "Tomato___Target_Spot",
             35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
             36: "Tomato___Tomato_mosaic_virus",
             37: "Tomato___healthy"}


class App:
    leaf_path = None
    leaf_prediction = ""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Predictions")
        self.width = 500
        self.height = 500
        self.window.resizable(False, False)
        self.window.geometry(str(self.width) + "x" + str(self.height))
        self.model = load_model("models/inception_01")

        button_load = tk.Button(
            self.window,
            text="Load Plant Image",
            command=lambda: self.load_action(),
            font=("Courier", 10))
        button_load.pack()
        button_load.place(width=200, height=40, x=150, y=20)

        self.window.mainloop()

    def load_action(self):
        self.leaf_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select a File",
            filetypes=(("JPG files", "*.JPG*"), ("all files", "*.*")))
        if not self.leaf_path:
            tk.messagebox.showerror(title="FileOpenError", message="File not Found")
            return

        self.process_image()
        return

    def predict(self, img_path):
        print("Predicting...")
        sample_image = image.load_img(img_path, target_size=(256, 256))
        input_arr = image.img_to_array(sample_image)
        input_arr = np.array([input_arr])  # Convert single image to a batch
        predictions = self.model.predict(input_arr).flatten()
        print(predictions)
        pred_string = class_map.get(np.argmax(predictions))
        return pred_string

    def process_image(self):
        image_to_display = Image.open(self.leaf_path)
        image_to_display = image_to_display.resize((256, 256))
        image_to_display = ImageTk.PhotoImage(image_to_display)
        # print(image_to_display.width(), "x", image_to_display.height())

        image_panel = tk.Label(image=image_to_display)
        image_panel.place(width=256, height=256, x=122, y=80)
        image_panel.img = image_to_display

        self.leaf_prediction = self.predict(self.leaf_path)
        leaf_label = tk.Label(self.window, text=self.leaf_prediction, font=("Courier", 14))
        leaf_label.place(width=500, height=50, x=0, y=350)
        return


def main():
    App()


if __name__ == '__main__':
    main()
