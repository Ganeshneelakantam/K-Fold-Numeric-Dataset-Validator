import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
# Add this import at the top of app.py
from flask import jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    # Additional data preprocessing steps can be added here
    return df

def validate_uploaded_file(uploaded_file):
    if not uploaded_file:
        flash('Error: No file uploaded.', 'error')
        return False
    if not allowed_file(uploaded_file.filename):
        flash('Error: Dataset must be in .csv format.', 'error')
        return False
    return True

def validate_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        flash('Error: No numeric columns in the dataset.', 'error')
        return False
    return True

@app.route('/')
def index():
    error_message = request.args.get('error_message')
    return render_template('index.html', error_message=error_message)


@app.route('/documentation')
def documentation():
    return render_template('document.html')
import traceback  # Import traceback module for detailed error logging

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        folds = int(request.form['folds'])
        kernel = request.form['kernel']
        random_state = int(request.form['random_state'])

        if not validate_uploaded_file(uploaded_file):
            return redirect(url_for('index'))

        try:
            df = pd.read_csv(uploaded_file)
            print("Dataset loaded successfully:", df.head())  # Print the loaded dataset for debugging
        except Exception as e:
            error_message = f'Error loading dataset: {e}'
            print("Error loading dataset:", traceback.format_exc())  # Log detailed error traceback
            return redirect(url_for('index', error_message=error_message))

        # Check if all columns are numeric
        if not df.select_dtypes(include=['number']).columns.equals(df.columns):
            error_message = 'Error: Dataset contains non-numeric data. Please check your dataset.'
            return redirect(url_for('index', error_message=error_message))

        try:
            df = preprocess_data(df)
            print("Dataset preprocessed successfully:", df.head())  # Print the preprocessed dataset for debugging
        except Exception as e:
            error_message = f'Error during preprocessing: {e}'
            print("Error during preprocessing:", traceback.format_exc())  # Log detailed error traceback
            return redirect(url_for('index', error_message=error_message))

        if not validate_numeric_columns(df):
            return redirect(url_for('index'))

        cross_validator = KFold(n_splits=folds, shuffle=True, random_state=42)

        algorithms = {
            'SVM': SVC(kernel=kernel, random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Neural Network': MLPClassifier(random_state=random_state)
        }

        X = df.select_dtypes(include=['number']).values
        y = df.iloc[:, -1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        results = {'individual_fold_accuracies': {}, 'algorithm_accuracies': {}, 'average_accuracy': {},
                   'num_folds': folds}

        for algorithm, model in algorithms.items():
            try:
                accuracies = cross_val_score(model, X, y, cv=cross_validator)
                average_accuracy = accuracies.mean()
                results['individual_fold_accuracies'][algorithm] = accuracies.tolist()
                results['algorithm_accuracies'][algorithm] = average_accuracy
                results['average_accuracy'][algorithm] = average_accuracy
            except Exception as e:
                error_message = f'Error during cross-validation for {algorithm}: {e}'
                print(f"Error during cross-validation for {algorithm}:", traceback.format_exc())  # Log detailed error traceback
                return redirect(url_for('index', error_message=error_message))

        dataset_head = df.head(10).to_dict()

        return render_template('results.html', results=results, dataset_head=dataset_head)

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=False)
