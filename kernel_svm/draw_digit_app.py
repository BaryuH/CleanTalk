import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle
from sklearn.svm import SVC

class DigitDrawerApp:
    def __init__(self, root, model=None):
        self.root = root
        self.root.title("MNIST Digit Drawer - SVM Prediction")
        self.model = model
        
        # Canvas settings
        self.canvas_size = 560  # 560 pixels on screen (20x magnification)
        self.grid_size = 28  # 28x28 actual grid
        self.pixel_size = 20  # Each grid cell = 20 screen pixels (bigger!)
        
        # Drawing data
        self.canvas_data = np.zeros((self.grid_size, self.grid_size))
        self.drawing = False
        
        # Create UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Draw a Digit (0-9)", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg='black', cursor='pencil')  # Black background like MNIST
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Right panel
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, padx=10, pady=10, sticky='n')
        
        # Prediction display
        pred_frame = ttk.LabelFrame(right_frame, text="Prediction", padding="20")
        pred_frame.grid(row=0, column=0, pady=10)
        
        self.pred_label = ttk.Label(pred_frame, text="?", font=("Arial", 72, "bold"), 
                                    foreground="gray")
        self.pred_label.grid(row=0, column=0)
        
        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=1, column=0, pady=10)
        
        self.predict_btn = ttk.Button(button_frame, text="🔮 Predict", command=self.predict)
        self.predict_btn.grid(row=0, column=0, pady=5, sticky='ew')
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=1, column=0, pady=5, sticky='ew')
        
        # Preview button to see what the model sees
        self.preview_btn = ttk.Button(button_frame, text="👁️ Show Model View", command=self.show_preview)
        self.preview_btn.grid(row=2, column=0, pady=5, sticky='ew')
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready to draw!", relief=tk.SUNKEN)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
    def start_draw(self, event):
        self.drawing = True
        self.draw(event)
        
    def stop_draw(self, event):
        self.drawing = False
        
    def draw(self, event):
        if self.drawing:
            # Calculate grid position
            x = event.x // self.pixel_size
            y = event.y // self.pixel_size
            
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Brush size matching MNIST stroke width (~2-3 pixels wide)
                brush_size = 1.5  # Creates ~3-4 pixel wide strokes like MNIST
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        xi, yj = x + i, y + j
                        if 0 <= xi < self.grid_size and 0 <= yj < self.grid_size:
                            # Gaussian-like falloff
                            distance = np.sqrt(i**2 + j**2)
                            if distance <= brush_size:
                                intensity = 1.0 - (distance / (brush_size + 1))
                                self.canvas_data[yj, xi] = min(1, self.canvas_data[yj, xi] + 0.6 * intensity)
                
                self.update_canvas()
                
    def update_canvas(self):
        self.canvas.delete('all')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.canvas_data[i, j] > 0:
                    # MNIST format: 0=black, 255=white
                    # canvas_data: 0-1 range → convert to 0-255 grayscale
                    gray = int(self.canvas_data[i, j] * 255)  # White digits on black bg
                    color = f'#{gray:02x}{gray:02x}{gray:02x}'
                    
                    x1 = j * self.pixel_size
                    y1 = i * self.pixel_size
                    x2 = x1 + self.pixel_size
                    y2 = y1 + self.pixel_size
                    
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        
        # Draw grid (darker for black background)
        for i in range(self.grid_size + 1):
            self.canvas.create_line(0, i * self.pixel_size, self.canvas_size, i * self.pixel_size, 
                                   fill='#333333')  # Dark gray grid
            self.canvas.create_line(i * self.pixel_size, 0, i * self.pixel_size, self.canvas_size, 
                                   fill='#333333')
    
    def predict(self):
        if self.model is None:
            self.status_label.config(text="Error: No model loaded!")
            self.pred_label.config(text="❌", foreground="red")
            return
            
        if np.sum(self.canvas_data) == 0:
            self.status_label.config(text="Please draw something first!")
            return
        
        # Preprocess to match MNIST format
        img = self.canvas_data.copy()
        
        # Normalize to 0-1 range (already done during drawing)
        # No additional normalization needed since we're already in 0-1 range
        
        # Reshape to match model input (1, 784)
        img = img.reshape(1, -1)
        
        # Make prediction
        try:
            prediction = self.model.predict(img)[0]
            
            # Get decision scores if available to show confidence
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(img)[0]
                confidence = np.max(scores)
                self.pred_label.config(text=str(prediction), foreground="green")
                self.status_label.config(text=f"Predicted: {prediction} (confidence: {confidence:.2f})")
            else:
                self.pred_label.config(text=str(prediction), foreground="green")
                self.status_label.config(text=f"Predicted: {prediction}")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.pred_label.config(text="❌", foreground="red")
    
    def clear_canvas(self):
        self.canvas_data = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete('all')
        self.update_canvas()
        self.pred_label.config(text="?", foreground="gray")
        self.status_label.config(text="Canvas cleared. Ready to draw!")
    
    def show_preview(self):
        """Show what the model actually sees"""
        if np.sum(self.canvas_data) == 0:
            self.status_label.config(text="Please draw something first!")
            return
        
        import tkinter.messagebox as messagebox
        
        # Create a text representation
        preview_text = "This is what the model sees:\n\n"
        preview_text += "28x28 pixel grid (white = 1.0, black = 0.0)\n\n"
        
        # Show a compressed view
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                avg = np.mean(self.canvas_data[i:i+2, j:j+2])
                if avg > 0.7:
                    preview_text += "█"
                elif avg > 0.4:
                    preview_text += "▓"
                elif avg > 0.2:
                    preview_text += "▒"
                elif avg > 0:
                    preview_text += "░"
                else:
                    preview_text += " "
            preview_text += "\n"
        
        preview_text += f"\nMax value: {np.max(self.canvas_data):.2f}"
        preview_text += f"\nMin value: {np.min(self.canvas_data):.2f}"
        preview_text += f"\nTotal pixels drawn: {np.sum(self.canvas_data > 0)}"
        
        messagebox.showinfo("Model Preview", preview_text)

def load_model_from_notebook():
    """Try to load model from saved file"""
    try:
        with open('mnist_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("⚠ No saved model found. Please train and save your model first.")
        print("\nAdd this to your notebook after training:")
        print("import pickle")
        print("with open('mnist_svm_model.pkl', 'wb') as f:")
        print("    pickle.dump(clf, f)")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("MNIST Digit Drawer - SVM Prediction App")
    print("=" * 50)
    
    # Load model
    model = load_model_from_notebook()
    
    # Create and run app
    root = tk.Tk()
    app = DigitDrawerApp(root, model)
    
    print("\n✓ App launched! Draw digits and click 'Predict'")
    print("Close the window to exit.\n")
    
    root.mainloop()
