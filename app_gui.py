import customtkinter as ctk
import os
import threading
from tkinter import filedialog
import subprocess

# Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ETLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MIMIC-IV Graph ETL Pipeline")
        self.geometry("600x500")

        # Title
        self.label = ctk.CTkLabel(self, text="Medical Graph Builder", font=("Roboto", 24))
        self.label.pack(pady=20)

        # 1. Base Path Selection
        self.path_frame = ctk.CTkFrame(self)
        self.path_frame.pack(pady=10, padx=20, fill="x")

        self.path_btn = ctk.CTkButton(self.path_frame, text="Select Data Folder", command=self.select_folder)
        self.path_btn.pack(side="left", padx=10)

        self.path_label = ctk.CTkLabel(self.path_frame, text="No folder selected")
        self.path_label.pack(side="left", padx=10)

        # 2. Batch Size Slider
        self.slider_frame = ctk.CTkFrame(self)
        self.slider_frame.pack(pady=10, padx=20, fill="x")

        self.slider_label = ctk.CTkLabel(self.slider_frame, text="Batch Size: 50")
        self.slider_label.pack(pady=5)

        self.slider = ctk.CTkSlider(self.slider_frame, from_=1, to=100, command=self.update_slider)
        self.slider.set(50)
        self.slider.pack(pady=5, fill="x", padx=20)

        # 3. API Key Input
        self.api_entry = ctk.CTkEntry(self, placeholder_text="Enter GROQ_API_KEY")
        self.api_entry.pack(pady=10, padx=20, fill="x")

        # 4. Console Log
        self.console = ctk.CTkTextbox(self, height=150)
        self.console.pack(pady=10, padx=20, fill="both", expand=True)

        # 5. Run Button
        self.run_btn = ctk.CTkButton(self, text="START PIPELINE", fg_color="green", command=self.start_thread)
        self.run_btn.pack(pady=20)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_label.configure(text=folder)
            os.environ["BASE_PATH"] = folder

    def update_slider(self, value):
        self.slider_label.configure(text=f"Batch Size: {int(value)}")

    def log(self, message):
        self.console.insert("end", message + "\n")
        self.console.see("end")

    def run_pipeline(self):
        # This function runs inside a separate thread
        self.log("--- Starting Process ---")

        # Example of how to call your script (You would integrate the logic directly here ideally)
        # For now, we simulate running the command
        self.log(f"Config: Batch={int(self.slider.get())}, Path={self.path_label.cget('text')}")
        self.log("Running ETL Script (00_load_batch_50.py)...")

        # In a real app, you would import main() from 00_load_batch_50 and call it.
        # Here is how you run the command:
        process = subprocess.Popen(
            ["python", "00_load_batch_50.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream output to the GUI console
        for line in process.stdout:
            self.log(line.strip())

        process.wait()
        self.log("--- Process Complete ---")
        self.run_btn.configure(state="normal")

    def start_thread(self):
        self.run_btn.configure(state="disabled")
        threading.Thread(target=self.run_pipeline, daemon=True).start()


if __name__ == "__main__":
    app = ETLApp()
    app.mainloop()