import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot UNO Interface")
        self.root.geometry("800x450")

        # Robot Moves
        self.rm = tk.Label(root, text="Robot Moves:", font=("Arial", 15))
        self.rm.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # Card Played
        self.tc = tk.Label(root, text="Top Card:", font=("Arial", 15))
        self.tc.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # Turn
        self.turn = tk.Label(root, text="Turn:", font=("Arial", 15))
        self.turn.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        # Image Placeholder
        self.image_label = tk.Label(root, text="Card Image", font=("Arial", 15), width=300, height=300, bg="gray")
        self.image_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def updateImage(self, card):
        try:
            photo = ImageTk.PhotoImage(Image.fromarray(card.astype(np.uint8)[...,::-1]))
            self.image_label.config(image=photo, text="")  # Remove text
            self.image_label.image = photo  # Keep reference
            self.root.update()
        except Exception as e:
            print("Image not found:", e)

    def updateMove(self, text):
        self.rm.config(text = f'Robot Move: {text}')
        self.root.update()

    def updateTopCard(self, text):
        self.tc.config(text = f'Top Card: {text}')
        self.root.update()

    def updateTurn(self, text):
        self.turn.config(text = f'Turn: {text}')
        self.root.update()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    counter = 0
    while True:
        counter += 1
        gui.turn.config(text = f'Test {counter}')
        gui.update_gui()
        root.update()
