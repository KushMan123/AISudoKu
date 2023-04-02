import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import cv2
from image_sudoku import process_image
from sudoku_cam import process_camera
from utilis import show_image

class ImageDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self. initUI()
        self.image_path=None
    
    def initUI(self):
        self.setWindowTitle("AI Sudoku Image")
        self.setGeometry(100,100,400,200)
        #Title Label Initialization and alignement
        self.title_label= QLabel("<h1>AI Sudoku Solver</h1>")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #Image Upload
        self.image_frame=QFrame(self)
        self.image_frame.setFrameShape(QFrame.Shape.HLine)
        self.image_section_title=QLabel("<h2> Image Section </h2>")
        self.image_path_lineEdit=QLineEdit()
        self.image_path_lineEdit.setObjectName("path_txt")
        self.image_browse_btn=QPushButton("...")
        self.image_browse_btn.setObjectName("browsw_btn")
        self.image_browse_btn.clicked.connect(self.select_image)      
        #Image Upload Layout
        self.image_upload_layout=QHBoxLayout()
        self.image_section=QVBoxLayout()
        self.image_upload_layout.addWidget(self.image_path_lineEdit)
        self.image_upload_layout.addWidget(self.image_browse_btn)
        self.image_section.addWidget(self.image_section_title)
        self.image_section.addLayout(self.image_upload_layout)
        
        #Solve Section
        self.solve_btn=QPushButton("Solve the Sudoku")
        self.solve_btn.clicked.connect(self.solve_sudoku)
        self.debug_check=QRadioButton("debug")
        self.sudoku_predict_title=QLabel("<h2>Predicted Board</h2>")
        self.sudoku_predict_board=QLabel("a\nb")
        self.sudoku_predict_board.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sudoku_board_layout=QVBoxLayout()
        self.sudoku_solve_btn_layout=QHBoxLayout()
        self.sudoku_section=QVBoxLayout()

        self.sudoku_board_layout.addWidget(self.sudoku_predict_title)
        self.sudoku_board_layout.addWidget(self.sudoku_predict_board)
        self.sudoku_solve_btn_layout.addWidget(self.debug_check)
        self.sudoku_solve_btn_layout.addWidget(self.solve_btn)
        self.sudoku_section.addLayout(self.sudoku_solve_btn_layout)
        self.sudoku_section.addLayout(self.sudoku_board_layout)

        #Window Layout
        self.window_layout=QVBoxLayout()
        self.window_layout.addWidget(self.title_label)
        self.window_layout.addWidget(self.image_frame)
        self.window_layout.addLayout(self.image_section)
        self.window_layout.addWidget(self.image_frame)
        self.window_layout.addLayout(self.sudoku_section)

        #Setting Window Layout
        self.wid=QWidget()
        self.setCentralWidget(self.wid)
        self.wid.setLayout(self.window_layout)

    def select_image(self):
        print("Button Clicked")
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "All Files (*);; PNG Files (*.png) ;; JPG (*.jpg);; JPEG(*.jpeg)",
        )
        self.image_path_lineEdit.setText(fname[0])
        self.image_path=fname[0]

    def solve_sudoku(self):
        debug_value=self.debug_check.isChecked()
        if (self.image_path_lineEdit.text()):
            img=cv2.imread(self.image_path)
            solution_found, img_out, predicted_board =process_image(img,debug_value)
            self.sudoku_predict_board.setText(predicted_board)
            show_image("Soduku Solution", img_out)
        

class SudokuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Sudoku Sover")
        self.setGeometry(100,100,300,100)

        #Label initialization and alignment
        title_label= QLabel("<h1>AI Sudoku Solver</h1>")
        title_text= QLabel("Choose how you like to get solution")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #Button Initialization and alignment
        image_btn=QPushButton("Image")
        realtime_btn=QPushButton("Realtime")
        realtime_btn.clicked.connect(self.solve_sudoku_cam)
        image_btn.clicked.connect(self.WhenImageClicked)

        #Layout setup
        title_layout=QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(title_text)
        button_layout=QHBoxLayout()
        button_layout.addWidget(image_btn)
        button_layout.addWidget(realtime_btn)
        window_layout=QVBoxLayout()
        window_layout.addLayout(title_layout)
        window_layout.addLayout(button_layout)

        #Setting up window layot
        wid=QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(window_layout)
        self.show()

    def WhenImageClicked(self):
        self.img_dialog=ImageDialog()
        self.img_dialog.show()

    def solve_sudoku_cam(self):
        process_camera()

def main():
    app=QApplication(sys.argv)
    window=SudokuWindow()
    sys.exit(app.exec())

if __name__=="__main__":
    main()