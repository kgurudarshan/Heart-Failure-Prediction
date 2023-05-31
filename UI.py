import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class HeartFailureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Failure Prediction")
        self.setGeometry(100, 100, 400, 200)
        
        self.dataset_loaded = False
        
        self.create_widgets()
        self.create_layout()
    
    def create_widgets(self):
        self.load_data_button = QPushButton("Load Dataset", self)
        self.load_data_button.clicked.connect(self.load_dataset)
        
        self.train_model_button = QPushButton("Train Model", self)
        self.train_model_button.setEnabled(False)
        self.train_model_button.clicked.connect(self.train_model)
        
        self.accuracy_label = QLabel("Accuracy:")
        self.confusion_matrix_label = QLabel("Confusion Matrix:")
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        
    def create_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.load_data_button)
        layout.addWidget(self.train_model_button)
        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.confusion_matrix_label)
        layout.addWidget(self.result_text)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def load_dataset(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv)")
        
        if filepath:
            self.dataset_loaded = True
            self.heart_data = pd.read_csv(filepath)
            self.train_model_button.setEnabled(True)
    
    def train_model(self):
        if self.dataset_loaded:
            X = self.heart_data.drop(['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp'],'columns')
            y = self.heart_data['output']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            self.accuracy_label.setText(f"Accuracy: {accuracy}")
            self.confusion_matrix_label.setText("Confusion Matrix:")
            self.result_text.setText(str(cm))
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartFailureApp()
    window.show()
    sys.exit(app.exec_())
