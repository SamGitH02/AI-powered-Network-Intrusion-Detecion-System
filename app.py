from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'  # Directory to save uploaded files

# Load the pre-trained model
model = load_model('model_trained_on_new_dataset.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part in the request", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        if file:
            # Save the file to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                # Load and preprocess the file
                data = pd.read_csv(file_path)
                data.fillna(0, inplace=True)
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data.fillna(0, inplace=True)

                # Debugging: Log column names after loading the file
                print("Columns in uploaded file:", data.columns.tolist())

                # Preserve Source IP and Destination IP columns
                source_ip = data['Source IP'] if 'Source IP' in data.columns else ['N/A'] * len(data)
                destination_ip = data['Destination IP'] if 'Destination IP' in data.columns else ['N/A'] * len(data)

                # Drop non-numeric columns except Source IP and Destination IP
                data = data.drop(columns=[col for col in data.select_dtypes(include=['object', 'category']).columns if col not in ['Source IP', 'Destination IP']], errors='ignore')

                # Ensure the 'Label' column is dropped
                if 'Label' in data.columns:
                    data = data.drop('Label', axis=1)

                # Debugging: Log column names and their count after preprocessing
                print("Columns after preprocessing:", data.columns.tolist())
                print("Number of columns after preprocessing:", data.shape[1])

                # Ensure Source IP and Destination IP are not dropped
                source_ip = data['Source IP'] if 'Source IP' in data.columns else ['N/A'] * len(data)
                destination_ip = data['Destination IP'] if 'Destination IP' in data.columns else ['N/A'] * len(data)

                # Select only the required 6 features for the model
                required_features = [
                    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Fwd Packet Length Max", "Bwd Packet Length Max", "Flow Bytes/s"
                ]
                missing_features = [feature for feature in required_features if feature not in data.columns]
                if missing_features:
                    return ("The uploaded file is missing the following required features: "
                            f"{', '.join(missing_features)}. Please ensure the file includes these features."), 400

                data = data[required_features]

                # Debugging: Log the values of the 6 required features
                print("Values of required features for prediction:", data[required_features].head())

                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data)

                # Validate the data size before reshaping
                expected_size = 6  # Model expects 6 features
                if X_scaled.shape[1] != expected_size:
                    return ("Uploaded file has an incorrect number of features. "
                            f"Expected {expected_size}, but got {X_scaled.shape[1]}. "
                            "Please ensure the file is preprocessed correctly and matches the required format."), 400

                # Reshape the data for the model
                X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, 6, 1)

                # Check if the reshaped data matches the model's input shape
                expected_shape = (1, 6, 1)
                if X_reshaped.shape[1:] != expected_shape:
                    return ("Uploaded file has an incorrect shape. "
                            f"Expected {expected_shape}, but got {X_reshaped.shape[1:]}. "
                            "Please ensure the file is preprocessed correctly and matches the required format."), 400

                # Perform predictions
                predictions = model.predict(X_reshaped)

                # Debugging: Log input features and predictions
                print("Input features for prediction:", X_scaled)
                print("Model predictions:", predictions)

                predicted_classes = np.argmax(predictions, axis=1)

                # Construct results for testing
                results = []
                for i, prediction in enumerate(predicted_classes):
                    results.append({
                        'Flow ID': f'Flow-{i+1}',
                        'Source IP': source_ip[i],
                        'Destination IP': destination_ip[i],
                        'Predicted_Label': 'Normal' if prediction == 0 else 'Attack',
                        'Confidence': f'{np.max(predictions[i]):.2f}'
                    })

                # Pass results to the template
                return render_template('results.html', results=results)

            except Exception as e:
                return f"Error processing file: {str(e)}", 500

    return render_template('analyze.html')

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        dataset_name = request.form['dataset']
        chart_type = request.form['chart_type']

        # Load the selected dataset
        dataset_path = os.path.join('CIC Dataset', dataset_name)
        if not os.path.exists(dataset_path):
            return "Dataset not found!"

        df = pd.read_csv(dataset_path)

        # Generate the requested chart
        plt.figure(figsize=(10, 6))
        if chart_type == 'bar':
            df['attack_detected'].value_counts().plot(kind='bar', color=['blue', 'orange'])
            plt.title('Attack Detection Distribution')
            plt.xlabel('Attack Detected')
            plt.ylabel('Count')
        elif chart_type == 'pie':
            df['attack_detected'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange'])
            plt.title('Attack Detection Distribution')
        else:
            return "Unsupported chart type!"

        # Save the plot to a file
        plot_path = os.path.join('static', 'visualization.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('visualization.html', plot_url=plot_path)

    return render_template('visualization.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)