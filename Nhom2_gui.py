import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

class YOLOAutoDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhóm 2 - Auto Detection with Line Counter")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.model = None
        self.current_image = None
        self.original_image = None
        self.model_path = None
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.video_cap = None
        self.detection_mode = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_thread = None
        self.playback_speed = 1.0
        self.fps = 0
        
        # Line drawing variables
        self.drawing_line = False
        self.line_start = None
        self.line_end = None
        self.counting_line = None
        self.crossed_ids = set()
        self.count_up = 0
        self.count_down = 0
        self.tracked_objects = {}
        
        # Total counting variables
        self.total_detected_ids = set()  # Track all unique IDs ever detected
        self.total_class_count = {}  # Total count per class across all frames
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e', height=60)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚦 Nhóm 2 - Traffic Detection & Counting", 
            font=('Arial', 22, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        title_label.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg='#1e1e1e', width=320)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.setup_controls(left_panel)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg='#1e1e1e')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_image_display(right_panel)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="⚠️ Chưa load model - Vui lòng chọn model để bắt đầu",
            bg='#1e1e1e',
            fg='#ff9800',
            font=('Arial', 11, 'bold'),
            anchor='w',
            padx=15,
            height=2
        )
        self.status_bar.pack(side='bottom', fill='x')
        
    def setup_controls(self, parent):
        # Model section
        model_frame = tk.LabelFrame(
            parent,
            text="📦 MODEL",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            padx=10,
            pady=10
        )
        model_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            model_frame,
            text="📁 Chọn Model (.pt)",
            command=self.load_model,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=12
        ).pack(fill='x', pady=5)
        
        self.model_label = tk.Label(
            model_frame,
            text="❌ Chưa có model",
            bg='#1e1e1e',
            fg='#ff9800',
            font=('Arial', 10, 'bold'),
            wraplength=280
        )
        self.model_label.pack(pady=5)
        
        # Input source section
        source_frame = tk.LabelFrame(
            parent,
            text="🎥 NGUỒN ĐẦU VÀO",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            padx=10,
            pady=10
        )
        source_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            source_frame,
            text="🖼️ Chọn Ảnh (Auto Detect)",
            command=self.load_image,
            bg='#2196F3',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=12
        ).pack(fill='x', pady=5)
        
        tk.Button(
            source_frame,
            text="🎬 Chọn Video (Auto Detect)",
            command=self.load_video,
            bg='#FF9800',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=12
        ).pack(fill='x', pady=5)
        
        self.webcam_btn = tk.Button(
            source_frame,
            text="📹 Webcam Realtime",
            command=self.toggle_webcam,
            bg='#9C27B0',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=12
        )
        self.webcam_btn.pack(fill='x', pady=5)
        
        # Line drawing section
        self.line_frame = tk.LabelFrame(
            parent,
            text="📏 VẼ ĐƯỜNG ĐẾM (VIDEO)",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            padx=10,
            pady=10
        )
        self.line_frame.pack(fill='x', padx=10, pady=10)
        self.line_frame.pack_forget()
        
        tk.Label(
            self.line_frame,
            text="Click 2 điểm trên video\nđể vẽ đường đếm",
            bg='#1e1e1e',
            fg='#FFD700',
            font=('Arial', 9),
            justify='center'
        ).pack(pady=5)
        
        self.draw_line_btn = tk.Button(
            self.line_frame,
            text="✏️ Vẽ Đường Đếm",
            command=self.start_drawing_line,
            bg='#00BCD4',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=15,
            pady=10
        )
        self.draw_line_btn.pack(fill='x', pady=5)
        
        tk.Button(
            self.line_frame,
            text="🗑️ Xóa Đường",
            command=self.clear_line,
            bg='#FF5722',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=15,
            pady=10
        ).pack(fill='x', pady=5)
        
        # Detection settings
        settings_frame = tk.LabelFrame(
            parent,
            text="⚙️ CÀI ĐẶT DETECTION",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            padx=10,
            pady=10
        )
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Confidence
        conf_frame = tk.Frame(settings_frame, bg='#1e1e1e')
        conf_frame.pack(fill='x', pady=5)
        
        tk.Label(
            conf_frame,
            text="Confidence:",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 10, 'bold')
        ).pack(side='left')
        
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_label = tk.Label(
            conf_frame,
            textvariable=self.conf_var,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Arial', 10, 'bold')
        )
        self.conf_label.pack(side='right')
        
        tk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient='horizontal',
            variable=self.conf_var,
            bg='#1e1e1e',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#424242',
            activebackground='#4CAF50',
            showvalue=False
        ).pack(fill='x', pady=(0, 10))
        
        # IOU
        iou_frame = tk.Frame(settings_frame, bg='#1e1e1e')
        iou_frame.pack(fill='x', pady=5)
        
        tk.Label(
            iou_frame,
            text="IOU Threshold:",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 10, 'bold')
        ).pack(side='left')
        
        self.iou_var = tk.DoubleVar(value=0.45)
        self.iou_label = tk.Label(
            iou_frame,
            textvariable=self.iou_var,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Arial', 10, 'bold')
        )
        self.iou_label.pack(side='right')
        
        tk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient='horizontal',
            variable=self.iou_var,
            bg='#1e1e1e',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#424242',
            activebackground='#4CAF50',
            showvalue=False
        ).pack(fill='x')
        
        # Results
        results_frame = tk.LabelFrame(
            parent,
            text="📊 KẾT QUẢ DETECTION",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            padx=10,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.results_text = tk.Text(
            results_frame,
            bg='#2b2b2b',
            fg='#00ff00',
            font=('Consolas', 10),
            wrap='word',
            relief='flat',
            padx=10,
            pady=10,
            yscrollcommand=scrollbar.set
        )
        self.results_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Control buttons
        control_frame = tk.Frame(parent, bg='#1e1e1e')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            control_frame,
            text="⏹️ Dừng",
            command=self.stop_detection,
            bg='#FF5722',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=10
        ).pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        tk.Button(
            control_frame,
            text="🗑️ Xóa",
            command=self.clear_all,
            bg='#f44336',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=20,
            pady=10
        ).pack(side='right', expand=True, fill='x', padx=(5, 0))
        
    def setup_image_display(self, parent):
        # Info bar
        info_frame = tk.Frame(parent, bg='#1e1e1e', height=40)
        info_frame.pack(fill='x', padx=10, pady=(10, 5))
        info_frame.pack_propagate(False)
        
        self.mode_label = tk.Label(
            info_frame,
            text="📷 Chế độ: Chưa chọn",
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Arial', 12, 'bold'),
            anchor='w'
        )
        self.mode_label.pack(side='left', padx=10)
        
        self.fps_label = tk.Label(
            info_frame,
            text="FPS: --",
            bg='#1e1e1e',
            fg='#FFD700',
            font=('Arial', 11, 'bold'),
            anchor='e'
        )
        self.fps_label.pack(side='right', padx=10)
        
        # Canvas
        canvas_frame = tk.Frame(parent, bg='#1e1e1e')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=(5, 5))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#2b2b2b',
            highlightthickness=2,
            highlightbackground='#00ff00'
        )
        self.canvas.pack(fill='both', expand=True)
        
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        self.canvas.create_text(
            500, 300,
            text="🚀 Chọn Model và Nguồn để bắt đầu\n\n"
                 "✓ Ảnh: Đếm tự động\n"
                 "✓ Video: Vẽ line để đếm\n"
                 "✓ Webcam: Đếm realtime",
            fill='#666666',
            font=('Arial', 14),
            tags='placeholder',
            justify='center'
        )
        
        # Video controls
        self.video_controls_frame = tk.Frame(parent, bg='#1e1e1e', height=120)
        self.video_controls_frame.pack(fill='x', padx=10, pady=(5, 10))
        self.video_controls_frame.pack_propagate(False)
        self.video_controls_frame.pack_forget()
        
        self.setup_video_controls()
        
    def setup_video_controls(self):
        tk.Label(
            self.video_controls_frame,
            text="🎬 VIDEO CONTROLS",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 11, 'bold')
        ).pack(pady=(10, 5))
        
        progress_frame = tk.Frame(self.video_controls_frame, bg='#1e1e1e')
        progress_frame.pack(fill='x', padx=15, pady=5)
        
        self.frame_label = tk.Label(
            progress_frame,
            text="Frame: 0 / 0",
            bg='#1e1e1e',
            fg='#FFD700',
            font=('Arial', 9, 'bold')
        )
        self.frame_label.pack(side='left')
        
        self.time_label = tk.Label(
            progress_frame,
            text="00:00 / 00:00",
            bg='#1e1e1e',
            fg='#FFD700',
            font=('Arial', 9, 'bold')
        )
        self.time_label.pack(side='right')
        
        self.progress_var = tk.DoubleVar()
        self.progress_scale = tk.Scale(
            self.video_controls_frame,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.progress_var,
            bg='#1e1e1e',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#424242',
            activebackground='#FF9800',
            showvalue=False,
            command=self.seek_video
        )
        self.progress_scale.pack(fill='x', padx=15, pady=5)
        
        btn_frame = tk.Frame(self.video_controls_frame, bg='#1e1e1e')
        btn_frame.pack(pady=5)
        
        tk.Button(
            btn_frame,
            text="⏪ -10s",
            command=lambda: self.skip_seconds(-10),
            bg='#424242',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=10,
            pady=5
        ).pack(side='left', padx=2)
        
        tk.Button(
            btn_frame,
            text="◀ -5s",
            command=lambda: self.skip_seconds(-5),
            bg='#424242',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=10,
            pady=5
        ).pack(side='left', padx=2)
        
        self.play_pause_btn = tk.Button(
            btn_frame,
            text="▶️ Play",
            command=self.toggle_play_pause,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=15,
            pady=5
        )
        self.play_pause_btn.pack(side='left', padx=5)
        
        tk.Button(
            btn_frame,
            text="+5s ▶",
            command=lambda: self.skip_seconds(5),
            bg='#424242',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=10,
            pady=5
        ).pack(side='left', padx=2)
        
        tk.Button(
            btn_frame,
            text="+10s ⏩",
            command=lambda: self.skip_seconds(10),
            bg='#424242',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            relief='flat',
            padx=10,
            pady=5
        ).pack(side='left', padx=2)
        
        speed_frame = tk.Frame(btn_frame, bg='#1e1e1e')
        speed_frame.pack(side='left', padx=10)
        
        tk.Label(
            speed_frame,
            text="Speed:",
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Arial', 9)
        ).pack(side='left', padx=(0, 5))
        
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.speed_var,
            values=["0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"],
            width=7,
            state='readonly',
            font=('Arial', 9)
        )
        speed_combo.pack(side='left')
        speed_combo.bind('<<ComboboxSelected>>', self.change_speed)
        
    def start_drawing_line(self):
        if self.detection_mode != 'video':
            messagebox.showwarning("Cảnh báo", "Chỉ vẽ line cho video!")
            return
        
        self.drawing_line = True
        self.line_start = None
        self.line_end = None
        self.draw_line_btn.config(text="⏸️ Đang vẽ...", bg='#FF9800')
        self.status_bar.config(text="✏️ Click 2 điểm để vẽ đường", fg='#FFD700')
        
    def on_canvas_click(self, event):
        if not self.drawing_line:
            return
        
        if self.line_start is None:
            self.line_start = (event.x, event.y)
            self.status_bar.config(text="✏️ Click điểm thứ 2")
        elif self.line_end is None:
            self.line_end = (event.x, event.y)
            self.finalize_line()
            
    def finalize_line(self):
        if self.line_start and self.line_end:
            self.counting_line = (*self.line_start, *self.line_end)
            self.drawing_line = False
            self.crossed_ids.clear()
            self.count_up = 0
            self.count_down = 0
            
            self.draw_line_btn.config(text="✏️ Vẽ Đường Đếm", bg='#00BCD4')
            self.status_bar.config(text="✅ Đã vẽ đường đếm!", fg='#4CAF50')
            
    def clear_line(self):
        self.counting_line = None
        self.line_start = None
        self.line_end = None
        self.crossed_ids.clear()
        self.count_up = 0
        self.count_down = 0
        self.drawing_line = False
        
        self.draw_line_btn.config(text="✏️ Vẽ Đường Đếm", bg='#00BCD4')
        self.status_bar.config(text="🗑️ Đã xóa đường đếm")
        
    def check_line_crossing(self, obj_id, center_x, center_y, prev_center):
        if not self.counting_line or not prev_center:
            return
        
        x1, y1, x2, y2 = self.counting_line
        prev_x, prev_y = prev_center
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        if intersect((prev_x, prev_y), (center_x, center_y), (x1, y1), (x2, y2)):
            if obj_id not in self.crossed_ids:
                self.crossed_ids.add(obj_id)
                
                cross_product = (x2 - x1) * (center_y - y1) - (y2 - y1) * (center_x - x1)
                prev_cross = (x2 - x1) * (prev_y - y1) - (y2 - y1) * (prev_x - x1)
                
                if prev_cross < 0 and cross_product > 0:
                    self.count_up += 1
                elif prev_cross > 0 and cross_product < 0:
                    self.count_down += 1
        
    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Chọn model YOLO",
                filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
            )
            
            if file_path:
                self.model_path = file_path
                self.model = YOLO(file_path)
                
                model_name = Path(file_path).name
                self.model_label.config(text=f"✅ {model_name}", fg='#4CAF50')
                self.status_bar.config(text=f"✅ Model: {model_name}", fg='#4CAF50')
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "="*50 + "\n")
                self.results_text.insert(tk.END, f"✅ MODEL: {model_name}\n")
                self.results_text.insert(tk.END, "="*50 + "\n\n")
                self.results_text.insert(tk.END, f"Classes: {len(self.model.names)}\n\n")
                for idx, name in self.model.names.items():
                    self.results_text.insert(tk.END, f"   {idx:2d}. {name}\n")
                    
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model:\n{str(e)}")
            
    def load_image(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng load model trước!")
            return
            
        try:
            file_path = filedialog.askopenfilename(
                title="Chọn ảnh",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
            )
            
            if file_path:
                self.stop_detection()
                self.video_controls_frame.pack_forget()
                self.line_frame.pack_forget()
                self.detection_mode = 'image'
                self.mode_label.config(text="📷 Chế độ: Ảnh")
                
                self.original_image = Image.open(file_path)
                self.status_bar.config(text="🔍 Đang detect...", fg='#FFD700')
                self.root.update()
                
                self.detect_image(self.original_image)
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load ảnh:\n{str(e)}")
    
    def load_video(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng load model trước!")
            return
            
        try:
            file_path = filedialog.askopenfilename(
                title="Chọn video",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.stop_detection()
                self.detection_mode = 'video'
                self.mode_label.config(text="🎬 Chế độ: Video")
                
                self.video_cap = cv2.VideoCapture(file_path)
                if not self.video_cap.isOpened():
                    messagebox.showerror("Lỗi", "Không thể mở video!")
                    return
                
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.current_frame = 0
                self.playback_speed = 1.0
                
                self.video_controls_frame.pack(fill='x', padx=10, pady=(5, 10))
                self.line_frame.pack(fill='x', padx=10, pady=10)
                self.progress_scale.config(to=self.total_frames)
                self.update_video_info()
                
                self.crossed_ids.clear()
                self.count_up = 0
                self.count_down = 0
                self.tracked_objects.clear()
                self.total_detected_ids.clear()
                self.total_class_count.clear()
                
                self.is_running = True
                self.is_paused = False
                self.play_pause_btn.config(text="⏸️ Pause", bg='#FF9800')
                
                self.status_bar.config(text="▶️ Đang phát video...", fg='#4CAF50')
                
                self.video_thread = threading.Thread(target=self.process_video, daemon=True)
                self.video_thread.start()
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load video:\n{str(e)}")
    
    def toggle_play_pause(self):
        if self.detection_mode == 'video' and self.video_cap is not None:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.play_pause_btn.config(text="▶️ Play", bg='#4CAF50')
                self.status_bar.config(text="⏸️ Tạm dừng", fg='#FFD700')
            else:
                self.play_pause_btn.config(text="⏸️ Pause", bg='#FF9800')
                self.status_bar.config(text="▶️ Đang phát...", fg='#4CAF50')
    
    def seek_video(self, value):
        if self.detection_mode == 'video' and self.video_cap is not None:
            frame_num = int(float(value))
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            self.update_video_info()
    
    def skip_seconds(self, seconds):
        if self.detection_mode == 'video' and self.video_cap is not None:
            frame_skip = int(seconds * self.fps)
            new_frame = max(0, min(self.current_frame + frame_skip, self.total_frames - 1))
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.current_frame = new_frame
            self.progress_var.set(new_frame)
            self.update_video_info()
    
    def change_speed(self, event=None):
        speed_str = self.speed_var.get()
        self.playback_speed = float(speed_str.replace('x', ''))
    
    def update_video_info(self):
        if self.detection_mode == 'video' and self.video_cap is not None:
            self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
            
            current_time = self.current_frame / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.time_label.config(text=f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def toggle_webcam(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng load model trước!")
            return
        
        if self.is_running and self.detection_mode == 'webcam':
            self.stop_detection()
        else:
            self.start_webcam()
    
    def start_webcam(self):
        try:
            self.stop_detection()
            self.video_controls_frame.pack_forget()
            self.line_frame.pack_forget()
            self.detection_mode = 'webcam'
            self.mode_label.config(text="📹 Chế độ: Webcam")
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở webcam!")
                return
            
            self.is_running = True
            self.webcam_btn.config(text="⏹️ Dừng Webcam", bg='#FF5722')
            self.status_bar.config(text="▶️ Webcam đang chạy...", fg='#4CAF50')
            
            thread = threading.Thread(target=self.process_webcam, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi webcam:\n{str(e)}")
    
    def process_webcam(self):
        frame_count = 0
        start_time = time.time()
        
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.detect_and_display_frame(frame_rgb, mode='webcam')
            
            if frame_count >= 30:
                frame_count = 0
                start_time = time.time()
    
    def process_video(self):
        frame_count = 0
        start_time = time.time()
        
        while self.is_running and self.video_cap is not None:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ret, frame = self.video_cap.read()
            if not ret:
                self.stop_detection()
                self.status_bar.config(text="✅ Video đã phát xong", fg='#4CAF50')
                break
            
            self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_var.set(self.current_frame)
            self.update_video_info()
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.detect_and_display_frame(frame_rgb, mode='video')
            
            if frame_count >= 30:
                frame_count = 0
                start_time = time.time()
            
            if self.playback_speed < 1.0:
                time.sleep((1.0 / 30) * (1.0 / self.playback_speed))
    
    def detect_and_display_frame(self, frame_rgb, mode='video'):
        try:
            results = self.model.track(
                frame_rgb,
                conf=self.conf_var.get(),
                iou=self.iou_var.get(),
                verbose=False,
                persist=True
            )
            
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            detected_objects = {}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = self.model.names[cls]
                    
                    detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
                    
                    obj_id = int(box.id[0].item()) if box.id is not None else None
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    if mode == 'video' and obj_id is not None:
                        # Track unique objects for total counting
                        if obj_id not in self.total_detected_ids:
                            self.total_detected_ids.add(obj_id)
                            self.total_class_count[class_name] = self.total_class_count.get(class_name, 0) + 1
                        
                        prev_center = None
                        if obj_id in self.tracked_objects:
                            prev_center = self.tracked_objects[obj_id]['center']
                        
                        self.tracked_objects[obj_id] = {
                            'center': (center_x, center_y),
                            'class': class_name
                        }
                        
                        if self.counting_line:
                            self.check_line_crossing(obj_id, center_x, center_y, prev_center)
                    
                    draw.rectangle([x1, y1, x2, y2], outline='#00ff00', width=2)
                    
                    label = f"{class_name} {conf:.2f}"
                    if obj_id is not None:
                        label = f"ID:{obj_id} {class_name} {conf:.2f}"
                    
                    bbox = draw.textbbox((x1, y1 - 20), label, font=font_small)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='#00ff00')
                    draw.text((x1, y1 - 20), label, fill='black', font=font_small)
            
            if mode == 'video' and self.counting_line:
                x1, y1, x2, y2 = self.counting_line
                draw.line([x1, y1, x2, y2], fill='#FF0000', width=4)
                draw.ellipse([x1-5, y1-5, x1+5, y1+5], fill='#FF0000')
                draw.ellipse([x2-5, y2-5, x2+5, y2+5], fill='#FF0000')
            
            img_width, img_height = img.size
            
            if mode == 'video':
                # Show total count for video (accumulated across all frames)
                if self.counting_line:
                    # With line: show line counter
                    counter_text = f"↑ UP: {self.count_up}  ↓ DOWN: {self.count_down}"
                    bbox = draw.textbbox((0, 0), counter_text, font=font)
                    box_width = bbox[2] - bbox[0] + 40
                    box_height = bbox[3] - bbox[1] + 30
                    
                    box_x = img_width - box_width - 20
                    box_y = img_height - box_height - 20
                    
                    draw.rectangle(
                        [box_x, box_y, box_x + box_width, box_y + box_height],
                        fill='#1e1e1e',
                        outline='#FF0000',
                        width=3
                    )
                    
                    draw.text((box_x + 20, box_y + 15), counter_text, fill='#FFD700', font=font)
                
                # Show total class count (below line counter or at bottom)
                if self.total_class_count:
                    try:
                        font_counter = ImageFont.truetype("arial.ttf", 14)
                    except:
                        font_counter = ImageFont.load_default()
                    
                    # Build counter text with all classes
                    class_lines = []
                    total_all = sum(self.total_class_count.values())
                    class_lines.append(f"📊 TOTAL: {total_all}")
                    
                    for class_name, count in sorted(self.total_class_count.items(), key=lambda x: x[1], reverse=True):
                        class_lines.append(f"{class_name}: {count}")
                    
                    counter_text_full = "\n".join(class_lines)
                    
                    # Calculate box size
                    lines = counter_text_full.split('\n')
                    max_width = 0
                    total_height = 0
                    
                    for line in lines:
                        bbox = draw.textbbox((0, 0), line, font=font_counter)
                        line_width = bbox[2] - bbox[0]
                        line_height = bbox[3] - bbox[1]
                        max_width = max(max_width, line_width)
                        total_height += line_height + 5
                    
                    box_width = max_width + 30
                    box_height = total_height + 20
                    
                    # Position at bottom right, above line counter if exists
                    box_x = img_width - box_width - 20
                    if self.counting_line:
                        box_y = img_height - box_height - 90  # Above line counter
                    else:
                        box_y = img_height - box_height - 20
                    
                    # Draw background box
                    draw.rectangle(
                        [box_x, box_y, box_x + box_width, box_y + box_height],
                        fill='#1e1e1e',
                        outline='#00ff00',
                        width=3
                    )
                    
                    # Draw text line by line
                    current_y = box_y + 10
                    for i, line in enumerate(lines):
                        color = '#FFD700' if i == 0 else '#00ff00'  # Gold for title, green for classes
                        draw.text((box_x + 15, current_y), line, fill=color, font=font_counter)
                        bbox = draw.textbbox((0, 0), line, font=font_counter)
                        current_y += (bbox[3] - bbox[1]) + 5
                        
            else:
                # For image/webcam: show current frame count only
                total_count = sum(detected_objects.values())
                counter_text = f"COUNT: {total_count}"
                bbox = draw.textbbox((0, 0), counter_text, font=font)
                box_width = bbox[2] - bbox[0] + 40
                box_height = bbox[3] - bbox[1] + 30
                
                box_x = img_width - box_width - 20
                box_y = img_height - box_height - 20
                
                draw.rectangle(
                    [box_x, box_y, box_x + box_width, box_y + box_height],
                    fill='#1e1e1e',
                    outline='#00ff00',
                    width=3
                )
                
                draw.text((box_x + 20, box_y + 15), counter_text, fill='#00ff00', font=font)
            
            self.current_image = img
            self.display_image(self.current_image)
            
            if hasattr(self, '_frame_counter'):
                self._frame_counter += 1
            else:
                self._frame_counter = 0
            
            if self._frame_counter % 10 == 0:
                self.update_results(detected_objects, mode)
            
        except Exception as e:
            print(f"Error: {e}")
    
    def detect_image(self, img):
        try:
            img_array = np.array(img)
            
            results = self.model.predict(
                img_array,
                conf=self.conf_var.get(),
                iou=self.iou_var.get(),
                verbose=False
            )
            
            annotated_img = img.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            detected_objects = {}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = self.model.names[cls]
                    
                    detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
                    
                    draw.rectangle([x1, y1, x2, y2], outline='#00ff00', width=3)
                    
                    label = f"{class_name} {conf:.2f}"
                    bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='#00ff00')
                    draw.text((x1, y1 - 25), label, fill='black', font=font)
            
            img_width, img_height = annotated_img.size
            total_count = sum(detected_objects.values())
            counter_text = f"COUNT: {total_count}"
            
            bbox = draw.textbbox((0, 0), counter_text, font=font)
            box_width = bbox[2] - bbox[0] + 40
            box_height = bbox[3] - bbox[1] + 30
            
            box_x = img_width - box_width - 20
            box_y = img_height - box_height - 20
            
            draw.rectangle(
                [box_x, box_y, box_x + box_width, box_y + box_height],
                fill='#1e1e1e',
                outline='#00ff00',
                width=3
            )
            
            draw.text((box_x + 20, box_y + 15), counter_text, fill='#00ff00', font=font)
            
            self.current_image = annotated_img
            self.display_image(self.current_image)
            self.update_results(detected_objects, 'image')
            
            self.status_bar.config(text=f"✅ Tổng: {total_count} objects", fg='#4CAF50')
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi detect:\n{str(e)}")
    
    def update_results(self, detected_objects, mode):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, "📊 KẾT QUẢ\n")
        self.results_text.insert(tk.END, "="*50 + "\n\n")
        
        if mode == 'video' and self.counting_line:
            self.results_text.insert(tk.END, "🚦 LINE COUNTER:\n")
            self.results_text.insert(tk.END, f"   ↑ Đi lên: {self.count_up}\n")
            self.results_text.insert(tk.END, f"   ↓ Đi xuống: {self.count_down}\n")
            self.results_text.insert(tk.END, f"   📊 Tổng qua line: {self.count_up + self.count_down}\n\n")
        
        if mode == 'video':
            # Show total accumulated count for video
            if self.total_class_count:
                total_all = sum(self.total_class_count.values())
                self.results_text.insert(tk.END, f"📦 TỔNG ĐÃ PHÁT HIỆN: {total_all} objects\n")
                self.results_text.insert(tk.END, f"   (Unique IDs: {len(self.total_detected_ids)})\n\n")
                self.results_text.insert(tk.END, "📋 Chi tiết theo class:\n")
                for obj, count in sorted(self.total_class_count.items(), key=lambda x: x[1], reverse=True):
                    self.results_text.insert(tk.END, f"   • {obj}: {count}\n")
            else:
                self.results_text.insert(tk.END, "❌ Chưa phát hiện object nào\n")
            
            # Show current frame detection
            if detected_objects:
                total_frame = sum(detected_objects.values())
                self.results_text.insert(tk.END, f"\n🎬 Frame hiện tại: {total_frame} objects\n")
                for obj, count in detected_objects.items():
                    self.results_text.insert(tk.END, f"   • {obj}: {count}\n")
        else:
            # For image/webcam: show current detection only
            if detected_objects:
                total = sum(detected_objects.values())
                self.results_text.insert(tk.END, f"📦 Phát hiện: {total}\n\n")
                for obj, count in sorted(detected_objects.items(), key=lambda x: x[1], reverse=True):
                    self.results_text.insert(tk.END, f"   • {obj}: {count}\n")
            else:
                self.results_text.insert(tk.END, "❌ Không phát hiện\n")
        
        self.results_text.insert(tk.END, f"\n{'='*50}\n")
        self.results_text.insert(tk.END, f"⚙️ Conf: {self.conf_var.get():.2f}\n")
        self.results_text.insert(tk.END, f"⚙️ IOU: {self.iou_var.get():.2f}\n")
    
    def display_image(self, img):
        self.canvas.delete('all')
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 1000
            canvas_height = 700
        
        img_copy = img.copy()
        img_copy.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img_copy)
        
        x = (canvas_width - img_copy.width) // 2
        y = (canvas_height - img_copy.height) // 2
        
        self.canvas.create_image(x, y, anchor='nw', image=self.photo)
    
    def stop_detection(self):
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        self.webcam_btn.config(text="📹 Webcam Realtime", bg='#9C27B0')
        
        if self.detection_mode in ['webcam', 'video']:
            self.status_bar.config(text="⏹️ Đã dừng", fg='#ff9800')
    
    def clear_all(self):
        self.stop_detection()
        self.canvas.delete('all')
        self.canvas.create_text(
            500, 300,
            text="🚀 Chọn Model và Nguồn\n\n"
                 "✓ Ảnh: Đếm tự động\n"
                 "✓ Video: Đếm theo frame\n"
                 "✓ Webcam: Đếm realtime",
            fill='#666666',
            font=('Arial', 14),
            tags='placeholder',
            justify='center'
        )
        self.current_image = None
        self.original_image = None
        self.detection_mode = None
        self.counting_line = None
        self.line_start = None
        self.line_end = None
        self.crossed_ids.clear()
        self.count_up = 0
        self.count_down = 0
        self.tracked_objects.clear()
        self.total_detected_ids.clear()
        self.total_class_count.clear()
        
        self.results_text.delete(1.0, tk.END)
        self.mode_label.config(text="📷 Chế độ: Chưa chọn")
        self.fps_label.config(text="FPS: --")
        self.status_bar.config(text="✅ Đã xóa", fg='#4CAF50')
        
        self.video_controls_frame.pack_forget()
        self.line_frame.pack_forget()

def main():
    root = tk.Tk()
    app = YOLOAutoDetectionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()