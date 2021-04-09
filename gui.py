import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image

def open_image(path):
    image = Image.open(path)
    return ImageTk.PhotoImage(image)

def predict(img_path):


    return "Some Pretty Leaf!"

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

    def process_image(self):
        image = ImageTk.PhotoImage(Image.open(self.leaf_path))
        image_panel = tk.Label(image=image)
        image_panel.place(width=256, height=256, x=122, y=80)
        image_panel.img = image

        self.leaf_prediction = predict(self.leaf_path)
        leaf_label = tk.Label(self.window, text=self.leaf_prediction, font=("Courier", 18))
        leaf_label.place(width=500, height=50, x=0, y=350)
        return

def main():
    App()

if __name__ == '__main__':
    main()
