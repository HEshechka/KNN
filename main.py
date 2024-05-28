import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from algorithm.KNN import KNN
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.stats import mode


class KNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN App")

        self.model = None

        # Variables to hold dataset and model data
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.k = 5

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Button to load dataset
        self.load_btn = tk.Button(self.root, text="Load Dataset", command=self.load_dataset)
        self.load_btn.pack(pady=5)

        # Button to train model
        self.train_btn = tk.Button(self.root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(pady=5)

        # Button to view accuracy
        self.accuracy_btn = tk.Button(self.root, text="View Accuracy", command=self.view_accuracy, state=tk.DISABLED)
        self.accuracy_btn.pack(pady=5)

        # Entries for custom prediction
        self.age_label = tk.Label(self.root, text="Enter Age:")
        self.age_label.pack(pady=5)
        self.age_entry = tk.Entry(self.root)
        self.age_entry.pack(pady=5)

        self.salary_label = tk.Label(self.root, text="Enter Salary:")
        self.salary_label.pack(pady=5)
        self.salary_entry = tk.Entry(self.root)
        self.salary_entry.pack(pady=5)

        self.predict_btn = tk.Button(self.root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)

    def load_dataset(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                if file_path.endswith("csv"):
                    self.dataset = pd.read_csv(file_path)
                    self.dataset = self.dataset.drop('Gender', axis=1)
                    sns.displot(self.dataset, x='Salary', hue='Purchase Iphone').savefig('results/baseinfo.png')
                    messagebox.showinfo("Info", "Dataset loaded successfully!")
                    plt.show()
                    self.train_btn.config(state=tk.NORMAL)
                else:
                    raise Exception('Not a csv file')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        if self.dataset is not None:
            X = self.dataset.drop('Purchase Iphone', axis=1)
            Y = self.dataset['Purchase Iphone']
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
            self.model = KNN(self.X_train, self.X_test, self.Y_train, self.Y_test, self.k)
            messagebox.showinfo("Info", "Model trained successfully!")

            plt.clf()
            self.visualize_knn_result()
            plt.xlabel('Age')
            plt.ylabel('Salary')
            plt.title('KNN Result')
            plt.legend()
            plt.savefig('results/knn_result.png')
            plt.show()

            self.accuracy_btn.config(state=tk.NORMAL)
            self.predict_btn.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Warning", "Please load the dataset first.")

    def view_accuracy(self):
        if self.X_train is not None:
            k_values = list(range(1, 15))
            accuracies = []
            for k in k_values:
                X = self.dataset.drop('Purchase Iphone', axis=1)
                Y = self.dataset['Purchase Iphone']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
                knn_model = KNN(X_train, X_test, Y_train, Y_test, k)
                accuracy = knn_model.accuracy()
                accuracies.append(accuracy)

            max_acc = 0.0
            k = 0

            for i in range(len(accuracies)):
                if accuracies[i] > max_acc and i % 2 == 0:
                    max_acc = accuracies[i]
                    k = i + 1

            if k != self.k:
                messagebox.showinfo('Find better k', f'Find better k value = {k} with accuracy {max_acc}. Start retraining model')
                self.k = k
                self.train_model()
            else:
                messagebox.showinfo('All ok', f'Your k value = {self.k}')
            plt.plot(k_values, accuracies, color='blue', marker='x', linestyle='dashed')
            plt.savefig('results/accuracy.png')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Please train the model first.")

    def predict(self):
        try:
            age = int(self.age_entry.get())
            salary = int(self.salary_entry.get())
            new_data = pd.DataFrame({'Age': [age], 'Salary': [salary]})
            knn_indices = self.model.predict_point(new_data)

            prediction = mode([self.Y_train.loc[idx] for idx in knn_indices])

            result = 'will purchase an iPhone' if prediction[0] == 1 else 'will not purchase an iPhone'
            messagebox.showinfo("Prediction Result", f'The person with age {age} and salary {salary} {result}.')

            plt.clf()
            self.visualize_knn_result()
            # Визуализируем новую точку
            plt.scatter(age, salary, color='red', label='New Point')

            # Визуализируем ближайших соседей
            colors = np.random.rand(len(knn_indices), 3)

            for i, idx in enumerate(knn_indices):
                # Получаем случайный цвет для данного соседа
                color = colors[i]

                # Определяем метку для данного соседа в зависимости от класса
                label = 'Neighbor (Not Purchased)' if self.Y_train.loc[idx] == 0 else 'Neighbor (Purchased)'

                # Отображаем точку с данными координатами и случайным цветом
                plt.scatter(self.model.X_train.loc[idx]['Age'], self.model.X_train.loc[idx]['Salary'], color=color,
                            label=label, marker='x')

            plt.xlabel('Age')
            plt.ylabel('Salary')
            plt.title('Prediction')
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {e}")

    def visualize_knn_result(self):
        try:
            plt.scatter(self.X_train[self.Y_train == 0]['Age'], self.X_train[self.Y_train == 0]['Salary'], color='blue',
                        label='Not Purchase')
            plt.scatter(self.X_train[self.Y_train == 1]['Age'], self.X_train[self.Y_train == 1]['Salary'],
                        color='orange', label='Purchase')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize KNN result: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()
