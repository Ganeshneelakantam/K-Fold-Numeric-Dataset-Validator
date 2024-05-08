# K-Fold Cross Validation Flask Application (Numeric-Dataset-Validator)

This Flask application performs K-Fold Cross Validation on uploaded numeric datasets and evaluates machine learning models. It provides insights into the dataset and model performance through visualizations and metrics.

## Project Structure

The project consists of the following files and directories:

- **app.py**: Contains the Flask application code for handling HTTP requests, loading datasets, performing cross-validation, and rendering HTML templates.
- **static/**: Directory containing static files such as CSS stylesheets, JavaScript, and images.
  - **data.gif**: GIF image used in the application interface.
  - **doc_style.css**: CSS stylesheet for documentation pages.
  - **predictive-chart.ico**: Icon file for the application.
  - **result.css**: CSS stylesheet for result pages.
  - **script.js**: JavaScript file for client-side interactions.
  - **style.css**: Main CSS stylesheet for styling the application.
- **templates/**: Directory containing HTML templates for different pages of the application.
  - **document.html**: Documentation page template.
  - **index.html**: Homepage template for uploading datasets and configuring cross-validation parameters.
  - **results.html**: Template for displaying cross-validation results, including dataset overview, individual fold accuracies, average accuracy, and visualizations.

## Usage

1. Clone the repository to your local machine:

    ```bash
    git clone <repository_url>
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:

    ```bash
    python app.py
    ```

4. Access the application in your web browser at `http://localhost:5000`.

## Features

- **Upload Dataset**: Users can upload structured numeric datasets in CSV format.
- **Configure Parameters**: Users can configure the number of folds, kernel type, and random state for cross-validation.
- **Cross-Validation**: Utilizes K-Fold Cross Validation to evaluate multiple machine learning algorithms.
- **Visualizations**: Provides line chart visualizations of algorithm accuracies.
- **Documentation**: Includes documentation explaining the application and its features.

## Dependencies

- Flask: Web framework for building the application.
- pandas: Library for data manipulation and analysis.
- scikit-learn: Library for machine learning algorithms and tools.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
