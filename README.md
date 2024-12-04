# ML Model Deployment with FastAPI and GCP Cloud Run

Personal project consisting of a production-level machine learning service that predicts the number of daily orders in an e-grocery company using FastAPI, containerized with Docker and deployed on Google Cloud Run. The data comes from a [Kaggle dataset](https://www.kaggle.com/c/rohlik-orders-forecasting-challenge/data).

## Project Structure

- `model-package/`: Contains the ML model training pipeline and package
  - Training pipeline components
  - Model validation
  - Feature engineering
  - Package publishing scripts

- `orders-api/`: FastAPI service that serves predictions
  - REST API endpoints
  - Input/output schemas
  - Configuration management
  - Logging setup

## Key Features

- Automated CI/CD pipeline using CircleCI
- Model versioning and package management via Gemfury
- Containerized deployment using Docker
- Automated testing with pytest and tox
- Type checking with mypy
- Production-ready logging configuration
- Health check endpoints

## API Endpoints

- `GET /api/v1/health`: Health check endpoint
- `POST /api/v1/predict`: Makes order volume predictions
  - Accepts JSON payload with order features
  - Returns predicted order volumes

## Development Setup

1. Install dependencies:
```bash
# Model package
cd model-package
pip install -r requirements/requirements.txt

# API
cd orders-api
pip install -r requirements/requirements.txt
```

2. Run tests:
```bash
# Model package
cd model-package
tox

# API
cd orders-api
tox
```

3. Run API locally:
```bash
cd orders-api
bash run.sh
```

## Deployment

The project uses CircleCI for automated deployments:

1. Tests the regression model
2. Publishes model package to Gemfury
3. Tests the FastAPI service
4. Builds and pushes Docker image to Google Artifact Registry
5. Deploys to Google Cloud Run

Required environment variables:
- `GEMFURY_PUSH_URL`: Gemfury package repository URL
- `GOOGLE_AUTH`: GCP service account key
- `GOOGLE_PROJECT_ID`: GCP project ID
- `GCP_REGION`: GCP region
- `ARTIFACT_REPO`: Artifact Registry repository name
- `IMAGE_NAME`: Docker image name
- `CLOUD_RUN_SERVICE`: Cloud Run service name

## Model Training

The regression model uses a custom Voting Regressor based on LightGBM. Cross-validation is performed using GroupKFold to prevent data leakage.

Feature engineering includes features such as:
- Temporal features (day of week, month, holidays)
- Location clustering
- Shopping intensity metrics

## License

MIT