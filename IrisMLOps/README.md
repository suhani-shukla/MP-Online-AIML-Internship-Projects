# 🌸 Iris MLOps API

An end-to-end **Machine Learning Operations (MLOps)** project that demonstrates the complete lifecycle of a machine learning application—from training a Scikit-learn Iris classifier to serving predictions through a Flask REST API, containerizing the application with Docker, and deploying it to Render.

---

## 🚀 Features

- Train an Iris classification model using Scikit-learn
- Save the trained model as a serialized artifact
- Expose predictions through a Flask REST API
- Validate API inputs
- Containerize the application with Docker
- Deploy to Render
- Test using cURL and Postman
- Production-ready project structure

---

## 📁 Project Structure

```text
IrisMLOps/
├── src/
│   ├── train.py           # Trains and saves the ML model
│   ├── app.py             # Flask REST API
│   └── schema.py          # Input validation schema
│
├── models/
│   ├── iris_model.pkl     # Trained model artifact
│   └── metrics.json       # Model evaluation metrics
│
├── tests/                 # Unit and integration tests
│
├── Dockerfile             # Docker configuration
├── .dockerignore          # Docker ignore rules
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.x |
| Machine Learning | Scikit-learn |
| Model | RandomForestClassifier |
| API Framework | Flask |
| WSGI Server | Gunicorn |
| Containerization | Docker |
| Deployment | Render |
| API Testing | Postman, cURL |
| Version Control | Git & GitHub |

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd IrisMLOps
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

This generates:

```text
models/
└── iris_model.pkl
```

### 5. Start the Flask server

```bash
python src/app.py
```

The API will be available at:

```
http://localhost:5000
```

---

# 🐳 Docker

## Build the Docker image

```bash
docker build -t iris-mlops-api .
```

## Run the container

```bash
docker run -e PORT=5000 -p 5000:5000 iris-mlops-api
```

The application will be available at:

```
http://localhost:5000
```

---

# ☁️ Live Deployment

The application is deployed on Render.

### Base URL

```
https://iris-mlops-ru47.onrender.com
```

---

# 📡 API Endpoints

## Health Check

**Request**

```http
GET /health
```

Example:

```
GET https://iris-mlops-ru47.onrender.com/health
```

Response

```json
{
  "status": "ok"
}
```

---

## Predict Flower Species

**Request**

```http
POST /predict
Content-Type: application/json
```

Example Request Body

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Successful Response

```json
{
  "class": "Setosa"
}
```

---

# 🧪 API Testing

## Using cURL

### Valid Request

```bash
curl -X POST https://iris-mlops-ru47.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.7,
    "sepal_width": 3.1,
    "petal_length": 4.7,
    "petal_width": 1.5
  }'
```

---

### Invalid Request (Missing Fields)

```bash
curl -i -X POST https://iris-mlops-ru47.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1}'
```

---

### Wrong HTTP Method

```bash
curl -i https://iris-mlops-ru47.onrender.com/predict
```

---

### Home Endpoint

```bash
curl https://iris-mlops-ru47.onrender.com/
```

---

# 📊 Model Information

- Dataset: Iris Dataset
- Algorithm: RandomForestClassifier
- Framework: Scikit-learn
- Train/Test Split: 80/20 (Stratified)
- Random State: 42
- Model Artifact: `models/iris_model.pkl`
- Evaluation Metrics: `models/metrics.json`

---

# ⚠️ Error Handling

| HTTP Status | Description |
|-------------|-------------|
| **200 OK** | Prediction successful |
| **400 Bad Request** | Missing or invalid input fields |
| **404 Not Found** | Invalid endpoint |
| **405 Method Not Allowed** | Incorrect HTTP method |
| **500 Internal Server Error** | Unexpected server error |

---

# 📦 Deployment Workflow

```text
Train Model
      │
      ▼
Save Model (.pkl)
      │
      ▼
Flask REST API
      │
      ▼
Docker Container
      │
      ▼
GitHub Repository
      │
      ▼
Render Deployment
      │
      ▼
Public REST API
```

---

# 🔮 Future Improvements

- GitHub Actions CI/CD pipeline
- Automated testing
- Model versioning with MLflow
- Prometheus monitoring
- Grafana dashboards
- Kubernetes deployment
- Horizontal autoscaling
- Logging and observability
- API authentication
- Input schema validation using Pydantic

---

# 👩‍💻 Author

**Suhani Shukla**

B.Tech Computer Science Engineering (Cyber Security & Digital Forensics)

---

# 📄 License

This project is developed for educational purposes as part of an end-to-end MLOps learning journey.