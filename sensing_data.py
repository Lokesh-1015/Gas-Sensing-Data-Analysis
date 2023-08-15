import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA as ICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.naive_bayes
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, \
    QLabel, QComboBox, QMessageBox, QFormLayout


class PCAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Gas Sensing')
        self.setGeometry(200, 200, 800, 600)

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create the file selection layout
        file_layout = QHBoxLayout()
        self.file_label = QLabel('No file selected')
        self.browse_button = QPushButton('Browse CSV File')
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.browse_button)

        # Create the column selection layout
        self.column_checkboxes = []
        self.column_layout = QHBoxLayout()

        # Create the PCA settings layout
        self.component_label = QLabel('Number of Components:')
        self.component_combo = QComboBox()
        self.component_combo.addItems(['1', '2', '3', '4', '5'])
        self.component_combo.setCurrentIndex(1)
        self.pca_button = QPushButton('Perform Dimensionality Reduction')
        self.pca_button.clicked.connect(self.perform_dimensionality_reduction)

        # Create the progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Create the results layout
        self.results_layout = QVBoxLayout()
        self.scatter_plot = QLabel()
        self.results_layout.addWidget(self.scatter_plot)
        self.principal_table = QTableWidget()
        self.results_layout.addWidget(self.principal_table)
        self.export_button = QPushButton('Export Results')
        self.export_button.clicked.connect(self.export_results)
        # self.results_layout.addWidget(self.export_button)

        # Create the model layout
        self.train_model_button = QPushButton('Train Model')
        self.train_model_button.clicked.connect(self.train_model)
        self.display_results_button = QPushButton('Display Results')
        self.display_results_button.clicked.connect(self.display_results)

        # Create the predict layout
        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predict_class)

        # Create the technique selection layout
        self.technique_label = QLabel('Technique:')
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(['PCA', 'LDA', 'ICA'])
        self.technique_combo.setCurrentIndex(0)

        # Create the technique settings layout
        self.settings_layout = QFormLayout()
        self.settings_layout.addRow(self.technique_label, self.technique_combo)

        # Create the best model layout
        self.train_model_button = QPushButton('Train Model')
        self.train_model_button.clicked.connect(self.train_model)
        self.display_results_button = QPushButton('Display Results')
        self.display_results_button.clicked.connect(self.display_results)
        self.best_model_button = QPushButton('Select Best Model')
        self.best_model_button.clicked.connect(self.select_best_model)

        # Create the model selection layout
        self.model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['RF', 'ANN', 'KNN', 'NB', 'SVM', 'XGboost'])
        self.model_combo.setCurrentIndex(0)

        # Create the dataset ratio layout
        self.ratio_label = QLabel('Dataset Ratio:')
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(['70:30', '80:20', '90:10'])
        self.ratio_combo.setCurrentIndex(0)

        # Add all layouts to the main layout
        main_layout.addLayout(file_layout)
        main_layout.addLayout(self.column_layout)
        main_layout.addWidget(self.component_label)
        main_layout.addWidget(self.component_combo)
        main_layout.addLayout(self.settings_layout)
        main_layout.addWidget(self.pca_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(self.results_layout)
        main_layout.addWidget(self.model_label)
        main_layout.addWidget(self.model_combo)
        main_layout.addWidget(self.ratio_label)
        main_layout.addWidget(self.ratio_combo)
        main_layout.addWidget(self.train_model_button)
        main_layout.addWidget(self.display_results_button)
        main_layout.addWidget(self.predict_button)
        main_layout.addWidget(self.best_model_button)

        # Create a central widget and set the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.df = None
        self.target_values = None
        self.selected_columns = None
        self.pca_result = None
        self.cluster_labels = None
        self.classifier = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('CSV Files (*.csv)')
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.file_label.setText(file_path)
                self.load_csv(file_path)

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path)

        # Clear any existing checkboxes
        for checkbox in self.column_checkboxes:
            checkbox.setParent(None)
        self.column_checkboxes.clear()

        # Create checkboxes for each column
        for column in self.df.columns:
            checkbox = QCheckBox(column)
            self.column_layout.addWidget(checkbox)
            self.column_checkboxes.append(checkbox)

    def perform_dimensionality_reduction(self):
        selected_columns = [checkbox.text() for checkbox in self.column_checkboxes if checkbox.isChecked()]
        n_components = int(self.component_combo.currentText())
        technique = self.technique_combo.currentText()

        if technique == 'PCA':
            # Perform PCA
            pca = PCA(n_components=n_components)
            self.pca_result = pca.fit_transform(self.df[selected_columns])
        elif technique == 'LDA':
            # Perform LDA
            lda = LDA(n_components=n_components)
            self.pca_result = lda.fit_transform(self.df[selected_columns], self.df[self.df.columns[-1]])
        elif technique == 'ICA':
            # Perform ICA
            ica = ICA(n_components=n_components)
            self.pca_result = ica.fit_transform(self.df[selected_columns])
        else:
            QMessageBox.warning(self, 'Error', 'Invalid technique selected.')
            return

        # Perform K-means clustering
        self.target_values = self.df[self.df.columns[-1]].unique()
        n_clusters = len(self.target_values)
        kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_labels = kmeans.fit_predict(self.df[selected_columns])

        # Update scatter plot based on user's choice
        if n_components == 2:  # 2D Scatter Plot
            plt.figure(figsize=(8, 6))
            for i, target_value in enumerate(self.target_values):
                plt.scatter(self.pca_result[self.cluster_labels == i, 0], self.pca_result[self.cluster_labels == i, 1],
                            label=target_value)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('2D {} Scatter Plot'.format(technique))
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif n_components >= 3:  # 3D Scatter Plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for i, target_value in enumerate(self.target_values):
                ax.scatter(self.pca_result[self.cluster_labels == i, 0], self.pca_result[self.cluster_labels == i, 1],
                           self.pca_result[self.cluster_labels == i, 2], label=target_value)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D {} Scatter Plot'.format(technique))
            ax.legend()
            plt.tight_layout()
            plt.show()

    def train_model(self):
        if self.pca_result is None or self.cluster_labels is None:
            QMessageBox.warning(self, 'Error', 'Perform PCA first.')
            return

        selected_columns = [checkbox.text() for checkbox in self.column_checkboxes if checkbox.isChecked()]

        # Split the data into train and test sets
        ratio_text = self.ratio_combo.currentText()
        train_ratio, test_ratio = map(int, ratio_text.split(':'))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.pca_result, self.cluster_labels, test_size=test_ratio / 100, train_size=train_ratio / 100,
            random_state=42
        )

        # Train the selected model
        model_text = self.model_combo.currentText()
        if model_text == 'RF':
            # Train Random Forest model
            model = RandomForestClassifier(n_jobs=-1)
        elif model_text == 'ANN':
            # Train Artificial Neural Network model
            model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
        elif model_text == 'KNN':
            # Train K-Nearest Neighbors model
            model = KNeighborsClassifier(n_neighbors=4)
        elif model_text == 'NB':
            # Train Naive Bayes model
            model = sklearn.naive_bayes.GaussianNB()
        elif model_text == 'SVM':
            # Train Support Vector Machine model
            model = svm.SVC(kernel='linear', C=1.0)
        elif model_text == 'xGboost':
            # Train XGBoost model
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)

        model.fit(self.X_train, self.y_train)

        self.classifier = model
        QMessageBox.information(self, 'Success', 'Model trained successfully.')

    def display_results(self):
        if self.classifier is None or self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, 'Error', 'Train the model first.')
            return

        # Make predictions on the test set
        y_pred = self.classifier.predict(self.X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # Create and display the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(np.arange(len(self.target_values)), self.target_values)
        plt.yticks(np.arange(len(self.target_values)), self.target_values)
        plt.tight_layout()
        plt.show()

        # Display the evaluation metrics
        report = classification_report(self.y_test, y_pred)
        metrics_message = f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}\n"
        QMessageBox.information(self, 'Evaluation Metrics', metrics_message + "\n\n" + report)

    def export_results(self):
        export_dialog = QFileDialog(self)
        export_dialog.setDefaultSuffix('csv')
        export_dialog.setAcceptMode(QFileDialog.AcceptSave)
        export_dialog.setNameFilter('CSV Files (*.csv)')
        if export_dialog.exec_():
            selected_files = export_dialog.selectedFiles()
            if selected_files:
                export_path = selected_files[0]
                self.df.to_csv(export_path, index=False)
                print(f'Results exported to {export_path}')

    def select_best_model(self):
        if self.pca_result is None or self.cluster_labels is None:
            QMessageBox.warning(self, 'Error', 'Perform PCA first.')
            return

        selected_columns = [checkbox.text() for checkbox in self.column_checkboxes if checkbox.isChecked()]

        # Split the data into train and test sets
        ratio_text = self.ratio_combo.currentText()
        train_ratio, test_ratio = map(int, ratio_text.split(':'))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.pca_result, self.cluster_labels, test_size=test_ratio / 100, train_size=train_ratio / 100,
            random_state=42
        )

        # Train and evaluate multiple models
        models = {
            'RF': RandomForestClassifier(),
            'ANN': MLPClassifier(),
            'KNN': KNeighborsClassifier(),
            'NB': sklearn.naive_bayes.GaussianNB(),
            'SVM': svm.SVC(),
            'XGBoost': XGBClassifier()
        }

        best_model = None
        best_accuracy = 0.0

        for model_name, model in models.items():
            model.fit(self.X_train, self.y_train)
            accuracy = model.score(self.X_test, self.y_test)
            if accuracy > best_accuracy:
                best_model = model_name
                best_accuracy = accuracy

        if best_model is not None:
            QMessageBox.information(self, 'Best Model',
                                    'Best model: {} (Accuracy: {:.2f}%)'.format(best_model, best_accuracy * 100))
        else:
            QMessageBox.warning(self, 'Error', 'No model found.')

    def predict_class(self):
        if self.classifier is None or self.pca_result is None:
            QMessageBox.warning(self, 'Error', 'Train the model and perform PCA first.')
            return

        selected_columns = [checkbox.text() for checkbox in self.column_checkboxes if checkbox.isChecked()]

        input_dialog = InputDialog(selected_columns)
        if input_dialog.exec_():
            input_values = input_dialog.get_input_values()
            if input_values:
                input_data = np.array([input_values], dtype=np.float64)
                pca_input = input_data[:, :self.pca_result.shape[1]]
                predicted_label = self.classifier.predict(pca_input)[0]
                predicted_label_name = self.target_values[predicted_label]  # Retrieve the label name
                QMessageBox.information(self, 'Prediction', f'The input belongs to the class: {predicted_label_name}')


class InputDialog(QDialog):
    def __init__(self, column_names):
        super().__init__()

        self.setWindowTitle('Input Data')
        self.setModal(True)

        self.column_names = column_names
        self.input_values = []

        main_layout = QVBoxLayout()

        input_layout = QFormLayout()

        for column_name in self.column_names:
            line_edit = QLineEdit()
            input_layout.addRow(column_name, line_edit)
            self.input_values.append(line_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addLayout(input_layout)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def get_input_values(self):
        input_values = []
        for line_edit in self.input_values:
            value = line_edit.text().strip()
            if value:
                input_values.append(value)
            else:
                QMessageBox.warning(self, 'Error', 'Please enter values for all fields.')
                return []
        return input_values


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCAWindow()
    window.show()
    sys.exit(app.exec_())