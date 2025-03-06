# Doodle Classifier

This project is a web-based doodle classifier that allows users to draw doodles on a canvas and classify them using a Convolutional Neural Network (CNN), a deep learning model.

## Project Structure

```
.gitignore
backend.py
modelD.h5
requirements.txt
templates/
    index.html
```

## Requirements

The project requires Python packages, which are listed in the `requirements.txt` file:

## Setup

1. Clone the repository:

```sh
git clone https://github.com/yourusername/doodle-classifier.git
cd doodle-classifier
```

2. Create a virtual environment and activate it:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

4. Place your model file (`modelD.h5`) in the root directory of the project.

## Running the Application

1. Start the Flask backend:

```sh
python backend.py
```

2. Open your web browser and navigate to `http://localhost:5000` to access the doodle classifier.

## Usage

1. Draw a doodle on the canvas.
2. Click the "Predict" button to classify the doodle.
3. The predicted class and confidence will be displayed below the canvas.
