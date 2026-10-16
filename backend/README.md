# ML Assistant Backend

FastAPI-based backend for the ML Assistant platform.

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ with pgvector
- Redis 7+
- Docker (optional, for containerized setup)

### Local Development

1. **Start PostgreSQL and Redis**
   ```bash
   docker-compose up -d
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Run migrations**
   ```bash
   alembic upgrade head
   ```

5. **Start the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start Celery worker (for async experiments)**
   ```bash
   celery -A backend.tasks worker --loglevel=info --concurrency=2
   ```

7. **Access API docs**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Compose (Full Stack)
```bash
docker-compose up -d
```
This starts PostgreSQL, Redis, Backend API, and Celery worker.

## Project Structure

```
backend/
├── alembic/           # Database migrations
├── api/               # API routes
│   ├── projects.py    # Project CRUD
│   ├── sessions.py    # Session management
│   ├── datasets.py    # Dataset upload/management
│   ├── experiments.py # Experiment orchestration
│   ├── models.py      # Model registry & comparison
│   └── ml.py          # ML pipeline (profiling, experiments)
├── config/            # Configuration management
├── database/          # Database models & connection
├── tasks/             # Celery tasks for async processing
├── main.py            # FastAPI application entry point
└── pyproject.toml     # Dependencies & tool config
```

## API Endpoints

### Projects
- `POST /projects` - Create project
- `GET /projects` - List projects (paginated)
- `GET /projects/{id}` - Get project
- `PATCH /projects/{id}` - Update project
- `DELETE /projects/{id}` - Archive project

### Sessions
- `POST /sessions?project_id=` - Create session
- `GET /sessions?project_id=` - List sessions
- `GET /sessions/{id}` - Get session
- `PATCH /sessions/{id}` - Update session
- `DELETE /sessions/{id}` - Delete session

### Datasets
- `POST /datasets?project_id=&session_id=` - Upload dataset
- `GET /datasets?project_id=` - List datasets
- `GET /datasets/{id}` - Get dataset
- `DELETE /datasets/{id}` - Delete dataset
- `POST /ml/datasets/{id}/profile` - Profile dataset (async with `async_mode=true`)

### ML Pipeline
- `POST /ml/experiments?project_id=` - Create experiment
- `POST /ml/experiments/{id}/run` - Run experiment (async with `async_mode=true`)
- `GET /ml/experiments/{id}/results` - Get experiment results with model ranking
- `GET /ml/experiments/{id}/best-model` - Get best model from experiment
- `GET /ml/tasks/{task_id}/status` - Check async task status
- `GET /ml/tasks/{task_id}/result` - Get async task result

### Models
- `POST /models?experiment_id=` - Register model
- `GET /models?experiment_id=` - List models
- `GET /models/{id}` - Get model
- `GET /models/experiment/{experiment_id}/comparison` - Compare models
- `DELETE /models/{id}` - Delete model

## Database Schema

- **projects** - Top-level containers for ML work
- **sessions** - Chat/work sessions within a project
- **datasets** - Uploaded data files with profiling info
- **experiments** - ML experiment runs with config & metrics
- **models** - Trained models with hyperparameters & performance

## ML Pipeline Features

- **Dataset Profiling**: Automatic data quality analysis, correlation detection, target analysis
- **Feature Engineering**: Automated preprocessing pipelines (imputation, encoding, scaling, outlier handling)
- **Model Training**: Multiple algorithms (Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, SVM, Neural Networks)
- **Cross-Validation**: Built-in CV for robust model evaluation
- **Model Ranking**: Automatic ranking by primary metric (F1 for classification, R2 for regression)
- **Async Processing**: Celery-based background jobs for long-running experiments

## Testing

```bash
cd backend
pytest tests/ -v
```

## Code Quality

```bash
# Format
ruff format .

# Lint
ruff check .

# Type check
mypy .
```