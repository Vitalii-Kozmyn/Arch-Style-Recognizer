import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import math

BG_COLOR = "#F2EBE5"      
FG_COLOR = "#1A3626"      
BTN_COLOR = "#2B5238"     
BTN_HOVER = "#1B3624"     
DEL_BTN_COLOR = "#A33333" 

MODEL_PATH = "src/arch_model_resnet18_vf_last.pth"

CLASS_NAMES = [
    "Achaemenid architecture",
    "American Foursquare architecture",
    "American craftsman style",
    "Ancient Egyptian architecture",
    "Art Deco architecture",
    "Art Nouveau architecture",
    "Baroque architecture",
    "Bauhaus architecture",
    "Beaux-Arts architecture",
    "Byzantine architecture",
    "Chicago school architecture",
    "Colonial architecture",
    "Deconstructivism",
    "Edwardian architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Greek Revival architecture",
    "International style",
    "Novelty architecture",
    "Palladian architecture",
    "Postmodern architecture",
    "Queen Anne architecture",
    "Romanesque architecture",
    "Tudor Revival architecture"
]

NUM_CLASSES = len(CLASS_NAMES)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    tensor = tensor.clone().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

def load_model():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Помилка", f"Модель не знайдена за шляхом: {MODEL_PATH}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)   
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

class ModernButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.default_bg = self["bg"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        if self["state"] != "disabled":
            if self.default_bg == BTN_COLOR:
                self["bg"] = BTN_HOVER
            elif self.default_bg == DEL_BTN_COLOR:
                self["bg"] = "#7A2424"

    def on_leave(self, e):
        self["bg"] = self.default_bg

class OutlineButton(tk.Frame):
    def __init__(self, master, text, command, padx=15, pady=8, **kw):
        super().__init__(master, bg=BTN_COLOR, padx=2, pady=2)
        self.btn = tk.Button(self, text=text, font=("Arial", 11, "bold"),
                             bg=BG_COLOR, fg=BTN_COLOR,
                             activebackground="#E2D7CE", activeforeground=BTN_COLOR,
                             relief="flat", borderwidth=0, cursor="hand2",
                             padx=padx, pady=pady, command=command, **kw)
        self.btn.pack(fill="both", expand=True)

        self.btn.bind("<Enter>", self.on_enter)
        self.btn.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.btn["bg"] = "#E2D7CE"

    def on_leave(self, e):
        self.btn["bg"] = BG_COLOR

class ArchitectureStylePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Визначення архітектурного стилю")
        self.root.minsize(650, 350)
        self.root.configure(bg=BG_COLOR)

        self.fields = []

        tk.Label(root, text="Оберіть зображення для аналізу:", font=("Arial", 16, "bold"), 
                 bg=BG_COLOR, fg=FG_COLOR).pack(pady=20)

        self.fields_frame = tk.Frame(root, bg=BG_COLOR)
        self.fields_frame.pack(pady=10)

        self.buttons_frame = tk.Frame(root, bg=BG_COLOR)
        self.buttons_frame.pack(pady=25)

        self.add_btn = OutlineButton(self.buttons_frame, text="➕ Додати ще зображення", command=self.add_field)
        self.add_btn.pack(side="left", padx=10)

        self.predict_btn = ModernButton(self.buttons_frame, text="🛠 Розпізнати стилі", 
                                        font=("Arial", 11, "bold"), bg=BTN_COLOR, fg="white", 
                                        activebackground=BTN_HOVER, activeforeground="white",
                                        relief="flat", borderwidth=0, padx=15, pady=8, cursor="hand2", 
                                        command=self.run_prediction)
        self.predict_btn.pack(side="left", padx=10)

        self.model_data = load_model()
        self.add_field()

    def add_field(self):
        row_frame = tk.Frame(self.fields_frame, bg=BG_COLOR)
        row_frame.pack(fill="x", pady=8)

        field_data = {"frame": row_frame, "path": None, "label": None}

        btn = ModernButton(row_frame, text="Вибрати зображення", width=22, 
                           font=("Arial", 10, "bold"), bg=BTN_COLOR, fg="white", 
                           activebackground=BTN_HOVER, activeforeground="white",
                           relief="flat", borderwidth=0, pady=5, cursor="hand2",
                           command=lambda: self.select_image(field_data))
        btn.pack(side="left", padx=15)

        lbl = tk.Label(row_frame, text="Файл не вибрано", font=("Arial", 10), 
                       bg=BG_COLOR, fg=FG_COLOR, width=40, anchor="w")
        lbl.pack(side="left")
        field_data["label"] = lbl

        del_btn = ModernButton(row_frame, text="✕", font=("Arial", 12, "bold"), 
                               bg=DEL_BTN_COLOR, fg="white", 
                               activebackground="#7A2424", activeforeground="white",
                               relief="flat", borderwidth=0, width=3, pady=2, cursor="hand2",
                               command=lambda: self.remove_field(field_data))
        del_btn.pack(side="left", padx=10)

        self.fields.append(field_data)

    def remove_field(self, field_data):
        field_data["frame"].destroy()
        self.fields.remove(field_data)

    def select_image(self, field_data):
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
        )
        if file_path:
            field_data["path"] = file_path
            field_data["label"].config(text=os.path.basename(file_path), fg=FG_COLOR)

    def run_prediction(self):
        if not self.model_data:
            return

        model, device = self.model_data
        valid_paths = [f["path"] for f in self.fields if f["path"] is not None]

        if not valid_paths:
            messagebox.showwarning("Увага", "Оберіть хоча б одне зображення!")
            return

        predictions = []
        tensors_to_show = []
        all_probabilities = []

        for path in valid_paths:
            try:
                img = Image.open(path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs[0], dim=0) * 100
                    _, preds = torch.max(outputs, 1)
                    predicted_class = CLASS_NAMES[preds[0].item()]

                predictions.append(predicted_class)
                tensors_to_show.append(transform(img))
                all_probabilities.append(probs.cpu().numpy())
            except Exception as e:
                messagebox.showerror("Помилка обробки", f"Не вдалося обробити файл {path}\n{e}")
                return

        self.show_results(valid_paths, tensors_to_show, predictions, all_probabilities)

    def show_results(self, paths, tensors, predictions, all_probs):
        num_images = len(paths)
        
        fig, axes = plt.subplots(2, num_images, figsize=(6 * num_images, 12))
        fig.canvas.manager.set_window_title('Результати розпізнавання стилів')
        
        fig.patch.set_facecolor(BG_COLOR)

        if num_images == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for i in range(num_images):
            ax_img = axes[0, i]
            display_img = denormalize(tensors[i])
            ax_img.imshow(display_img)
            ax_img.set_title(f"Стиль: {predictions[i]}", fontsize=14, color=FG_COLOR, fontweight="bold", pad=10)
            ax_img.axis('off')

            ax_tbl = axes[1, i]
            ax_tbl.axis('tight')
            ax_tbl.axis('off')

            sorted_indices = np.argsort(all_probs[i])[::-1]
            table_data = [[CLASS_NAMES[idx], f"{all_probs[i][idx]:.2f}%"] for idx in sorted_indices]

            table = ax_tbl.table(cellText=table_data, colLabels=["Стиль", "Збіг (%)"], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.2)
            
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor(BG_COLOR)
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(BTN_COLOR)
                else:
                    cell.set_facecolor('white')
                    cell.set_text_props(color=FG_COLOR)

        plt.tight_layout(pad=4.0, h_pad=5.0, w_pad=3.0)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ArchitectureStylePredictor(root)
    root.mainloop()