import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import os

class DividendPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dividend Prediction App")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Configure styles
        self.configure_styles()
        
        # Load the trained model
        self.model = None
        self.load_model()
        
        # Create landing screen
        self.create_landing_screen()
    
    def configure_styles(self):
        """Configure ttk styles for better appearance"""
        style = ttk.Style()
        
        # Configure main colors
        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.text_color = "#34495e"
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
        
        # Configure frame styles
        style.configure("Main.TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background="white", relief="solid", borderwidth=1)
        
        # Configure label styles
        style.configure("Title.TLabel", 
                       font=("Segoe UI", 24, "bold"), 
                       foreground=self.primary_color,
                       background=self.bg_color)
        
        style.configure("Subtitle.TLabel", 
                       font=("Segoe UI", 12), 
                       foreground="#7f8c8d",
                       background=self.bg_color)
        
        style.configure("Desc.TLabel", 
                       font=("Segoe UI", 10), 
                       foreground=self.text_color,
                       background=self.bg_color,
                       wraplength=400)
        
        style.configure("Feature.TLabel", 
                       font=("Segoe UI", 11, "bold"), 
                       foreground=self.primary_color,
                       background=self.bg_color)
        
        # Configure button styles
        style.configure("Accent.TButton",
                       font=("Segoe UI", 12, "bold"),
                       padding=10,
                       background=self.success_color,
                       foreground="white",
                       borderwidth=0,
                       relief="flat",
                       focuscolor="none")
        
        style.map("Accent.TButton",
                  background=[("active", self.success_color), ("!active", self.success_color)],
                  foreground=[("active", "white"), ("!active", "white")])
        
        style.configure("Secondary.TButton",
                       font=("Segoe UI", 11, "bold"),
                       padding=8,
                       background=self.success_color,
                       foreground="white",
                       borderwidth=0,
                       relief="flat",
                       focuscolor="none")
        
        style.map("Secondary.TButton",
                  background=[("active", self.success_color), ("!active", self.success_color)],
                  foreground=[("active", "white"), ("!active", "white")])
        
        # Configure entry styles
        style.configure("Custom.TEntry",
                       fieldbackground="white",
                       borderwidth=2,
                       relief="solid",
                       padding=5)
        
        style.map("Custom.TEntry",
                  fieldbackground=[("focus", "#ecf0f1")],
                  bordercolor=[("focus", self.accent_color)])
        
        # Configure separator
        style.configure("Custom.TSeparator", background="#bdc3c7")
        
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = "models/dividend_prediction_model_lr_tuned.pkl"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                messagebox.showerror("Error", f"Model file not found: {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def create_landing_screen(self):
        """Create the landing screen with enhanced styling"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
        
        # Create a card-like container with proper sizing
        card_frame = ttk.Frame(self.root, style="Card.TFrame")
        card_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=450)
        
        # Main frame inside card
        main_frame = ttk.Frame(card_frame, padding="30", style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Dividend Prediction System",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        # Decorative separator
        separator = ttk.Separator(main_frame, orient="horizontal", style="Custom.TSeparator")
        separator.pack(fill="x", pady=(0, 20))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Predict dividend changes using advanced machine learning",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Description with card effect
        desc_frame = ttk.Frame(main_frame, style="Card.TFrame")
        desc_frame.pack(pady=(0, 20), padx=10, fill="both", expand=True)
        
        desc_text = """This application uses machine learning to predict whether 
a company's dividend will increase or decrease based on:

• Price Momentum (YoY price change)
• Price Range Ratio (High/Low ratio)  
• Volume Momentum (YoY volume change)

The model has been trained on historical data from 
39 major technology companies (2011-2023)."""
        
        desc_label = ttk.Label(
            desc_frame,
            text=desc_text,
            style="Desc.TLabel",
            justify=tk.CENTER
        )
        desc_label.pack(pady=15, padx=15)
        
        # Button frame
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.pack(pady=10)
        
        # Start button
        start_button = tk.Button(
            button_frame,
            text="Start Prediction",
            command=self.create_prediction_screen,
            bg="#27ae60",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=10,
            relief="flat",
            bd=0,
            activebackground="#229954",
            activeforeground="white"
        )
        start_button.pack(pady=5)
        
        # Exit button
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.root.quit,
            bg="#27ae60",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=8,
            relief="flat",
            bd=0,
            activebackground="#229954",
            activeforeground="white"
        )
        exit_button.pack(pady=5)
    
    def create_prediction_screen(self):
        """Create the prediction screen with enhanced styling"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
        
        # Create a card-like container with proper sizing
        card_frame = ttk.Frame(self.root, style="Card.TFrame")
        card_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=450)
        
        # Main frame inside card
        main_frame = ttk.Frame(card_frame, padding="30", style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Enter Financial Indicators",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        # Decorative separator
        separator = ttk.Separator(main_frame, orient="horizontal", style="Custom.TSeparator")
        separator.pack(fill="x", pady=(0, 20))
        
        # Feature input fields
        features = [
            ("Price Momentum", "Enter price momentum (e.g., 0.25 for 25% YoY change)"),
            ("Price Range Ratio", "Enter price range ratio (e.g., 1.8 for High/Low)"),
            ("Volume Momentum", "Enter volume momentum (e.g., 0.10 for 10% YoY change)")
        ]
        
        self.entries = {}
        
        # Create a frame for inputs
        input_frame = ttk.Frame(main_frame, style="Main.TFrame")
        input_frame.pack(fill="both", expand=True)
        
        for i, (feature_name, placeholder) in enumerate(features):
            # Create a subframe for each input row
            row_frame = ttk.Frame(input_frame, style="Main.TFrame")
            row_frame.pack(fill="x", pady=10)
            
            # Label
            label = ttk.Label(
                row_frame,
                text=f"{feature_name}:",
                style="Feature.TLabel",
                width=20,
                anchor="w"
            )
            label.pack(side="left", padx=(0, 10))
            
            # Entry with custom style
            entry = ttk.Entry(row_frame, width=30, style="Custom.TEntry", font=("Segoe UI", 10))
            entry.pack(side="left")
            
            # Placeholder
            entry.insert(0, placeholder)
            entry.config(foreground="gray")
            
            # Bind events for placeholder
            entry.bind("<FocusIn>", lambda e, widget=entry, text=placeholder: self.on_entry_focus_in(e, widget, text))
            entry.bind("<FocusOut>", lambda e, widget=entry, text=placeholder: self.on_entry_focus_out(e, widget, text))
            
            self.entries[feature_name] = entry
        
        # Buttons frame with styling
        buttons_frame = ttk.Frame(main_frame, style="Main.TFrame")
        buttons_frame.pack(pady=20)
        
        # Predict button
        predict_button = tk.Button(
            buttons_frame,
            text="Predict",
            command=self.make_prediction,
            bg="#27ae60",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=10,
            relief="flat",
            bd=0,
            activebackground="#229954",
            activeforeground="white"
        )
        predict_button.pack(side="left", padx=5)
        
        # Back button
        back_button = tk.Button(
            buttons_frame,
            text="Back to Home",
            command=self.create_landing_screen,
            bg="#27ae60",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=8,
            relief="flat",
            bd=0,
            activebackground="#229954",
            activeforeground="white"
        )
        back_button.pack(side="left", padx=5)
    
    def on_entry_focus_in(self, event, widget, placeholder):
        """Handle entry focus in event"""
        if widget.get() == placeholder:
            widget.delete(0, tk.END)
            widget.config(foreground="black")
    
    def on_entry_focus_out(self, event, widget, placeholder):
        """Handle entry focus out event"""
        if widget.get() == "":
            widget.insert(0, placeholder)
            widget.config(foreground="gray")
    
    def make_prediction(self):
        """Make prediction using the model"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please check if the model file exists.")
            return
        
        try:
            # Get input values
            price_momentum = self.entries["Price Momentum"].get()
            price_range_ratio = self.entries["Price Range Ratio"].get()
            volume_momentum = self.entries["Volume Momentum"].get()
            
            # Validate inputs
            if (price_momentum == "Enter price momentum (e.g., 0.25 for 25% YoY change)" or 
                price_range_ratio == "Enter price range ratio (e.g., 1.8 for High/Low)" or 
                volume_momentum == "Enter volume momentum (e.g., 0.10 for 10% YoY change)"):
                messagebox.showwarning("Input Required", "Please fill in all fields.")
                return
            
            # Convert to float
            price_momentum = float(price_momentum)
            price_range_ratio = float(price_range_ratio)
            volume_momentum = float(volume_momentum)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Price_Momentum': [price_momentum],
                'Price_Range_Ratio': [price_range_ratio],
                'Volume_Momentum': [volume_momentum]
            })
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]
            
            # Format result
            if prediction == 1:
                result = "Dividend INCREASE"
                confidence = probability[1] * 100
                color = "green"
            else:
                result = "No Dividend Increase"
                confidence = probability[0] * 100
                color = "red"
            
            # Show result
            message = f"""Prediction Result: {result}

Confidence: {confidence:.1f}%

Input Values:
• Price Momentum: {price_momentum:.4f}
• Price Range Ratio: {price_range_ratio:.4f}
• Volume Momentum: {volume_momentum:.4f}

Note: This prediction is based on historical data from technology companies and should not be used as financial advice."""
            
            messagebox.showinfo("Prediction Result", message)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = DividendPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()