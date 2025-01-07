import tkinter as tk
from tkinter import Canvas, Label, Button, filedialog
import numpy as np
from PIL import Image, ImageGrab
import tensorflow as tf
import os
from datetime import datetime

# Load your trained MNIST model
model = tf.keras.models.load_model("mnist.h5")  # Replace with your model path

# Define the GUI class
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        
        # Canvas for drawing
        self.canvas = Canvas(master, width=280, height=280, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Buttons
        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=0, pady=10)
        
        self.clear_button = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, pady=10)
        

        
        # Labels for results
        self.result_label = Label(master, text="Prediction: ", font=("Helvetica", 16))
        self.result_label.grid(row=2, column=0, columnspan=2)
        
        self.accuracy_label = Label(master, text="Accuracy: ", font=("Helvetica", 16))
        self.accuracy_label.grid(row=3, column=0, columnspan=2)
    
    def draw(self, event):
        """Draw on the canvas."""
        x, y = event.x, event.y
        radius = 4
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black", outline="black")
    
    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.result_label.config(text="Prediction: ")
        self.accuracy_label.config(text="Confidence: ")
    
    def predict_digit(self):
        """Predict the digit from the drawn image."""
        # Grab the canvas content
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))
        
        # Preprocess the image
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
        
        # Predict the digit
        predictions = model.predict(img_array)
        digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Display the result
        self.result_label.config(text=f"Prediction: {digit}")
        self.accuracy_label.config(text=f"Confidence: {confidence:.2%}")
    


# Create the Tkinter window
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
