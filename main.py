import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from algorithm.KNN import KNN
import tkinter as tk
from tkinter import filedialog, messagebox


class KNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN App")

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
            y_hat_test = KNN(self.X_train, self.X_test, self.Y_train, self.Y_test, k_val=self.k)
            messagebox.showinfo("Info", "Model trained successfully!")
            self.knn_results(y_hat_test)
            self.accuracy_btn.config(state=tk.NORMAL)
            self.predict_btn.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Warning", "Please load the dataset first.")

    def view_accuracy(self):
        if self.X_train is not None:
            accuracy_vals = []
            for i in range(1, 15):
                y_hat_test = KNN(self.X_train, self.X_test, self.Y_train, self.Y_test, k_val=i)
                accuracy_vals.append(accuracy_score(self.Y_test, y_hat_test))
            k = accuracy_vals.index(max(accuracy_vals)) + 1
            if k != self.k:
                messagebox.showinfo('Find better k', f'Find better k value = {k} (your k now = {self.k}). Start retraining model')
                self.k = k
                self.train_model()
            else:
                messagebox.showinfo('All ok', f'Your k value = {self.k}')
            plt.plot(range(1, 15), accuracy_vals, color='blue', marker='x', linestyle='dashed')
            plt.savefig('results/accuracy.png')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Please train the model first.")

    def predict(self):
        try:
            age = int(self.age_entry.get())
            salary = int(self.salary_entry.get())
            new_data = pd.DataFrame({'Age': [age], 'Salary': [salary]})
            prediction = KNN(self.X_train, new_data, self.Y_train, self.Y_train, k_val=5)
            result = 'will purchase an iPhone' if prediction[0] == 1 else 'will not purchase an iPhone'
            messagebox.showinfo("Prediction Result", f'The person with age {age} and salary {salary} {result}.')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {e}")

    def knn_results(self, y_hat_test):
        plt.clf()
        for i in range(len(y_hat_test)):
            if y_hat_test[i] == 0:
                plt.scatter(self.X_test.iloc[i]['Age'], self.X_test.iloc[i]['Salary'], color='blue')
            if y_hat_test[i] == 1:
                plt.scatter(self.X_test.iloc[i]['Age'], self.X_test.iloc[i]['Salary'], color='orange')
        sns.scatterplot(data=self.dataset, x=self.X_test['Age'], y=self.X_test['Salary'], hue=self.Y_test)
        plt.xlabel('Age')
        plt.ylabel('Salary')
        plt.title('KNN Result')
        plt.savefig('results/knn_result.png')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()
