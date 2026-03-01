# Forecast Engine

A full-stack sales forecasting and business analytics platform built for local businesses like restaurants and bars. Uses machine learning to predict demand patterns by combining historical sales data with external signals — weather, local sports events, holidays, and more.

## What It Does

A bar owner shouldn't need a data science degree to know they need extra inventory for a sunny Saturday during football season. This platform automates that.

- **Demand Forecasting** — XGBoost-powered predictions that learn from your sales history and external factors
- **External Data Integration** — Automatically pulls weather forecasts, local sports schedules, and holiday calendars to improve predictions
- **Inventory Recommendations** — Translates demand forecasts into actionable prep/ordering suggestions
- **Auto-Comparison Testing** — Built-in model evaluation to compare forecast accuracy across different configurations
- **Scheduled Retraining** — Background scheduler keeps models fresh as new data comes in

## Tech Stack

### Backend
- Python / FastAPI
- SQLAlchemy + Alembic (database migrations)
- XGBoost (ML model)
- APScheduler (background jobs)
- JWT Authentication

### Frontend
- HTML / CSS / JavaScript
- Interactive dashboards for forecast visualization

### Infrastructure
- PostgreSQL
- Render (deployment via render.yaml)
- Environment-based configuration

## Project Structure

```
├── ai/                  # ML models, training pipelines, feature engineering
├── alembic/             # Database migration scripts
├── frontend/            # Dashboard UI
├── routers/             # API route handlers
├── services/            # Business logic layer
├── utils/               # Shared utilities
├── auth.py              # JWT authentication
├── config.py            # App configuration
├── database.py          # DB connection management
├── db_models.py         # SQLAlchemy ORM models
├── main.py              # FastAPI application entry point
├── processor.py         # Data processing pipeline
├── scheduler.py         # Background job scheduling
├── schemas.py           # Pydantic request/response models
└── settings.py          # Environment settings
```

## How It Works

1. **Data Ingestion** — Business uploads or connects historical sales data (date, item, quantity, revenue)
2. **Feature Engineering** — System enriches each data point with external features: was it raining? Was there a Broncos game? Was it a holiday weekend?
3. **Model Training** — XGBoost model trains on enriched dataset, learning which external factors actually move the needle for your specific business
4. **Forecasting** — Generates forward-looking demand predictions using upcoming weather forecasts and event schedules
5. **Recommendations** — Converts predictions into plain-language inventory and staffing suggestions

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL
- API keys for weather and sports data providers

### Setup

```bash
# Clone the repo
git clone https://github.com/mivashinko88-netizen/forecasting.git
cd forecasting

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database URL and API keys

# Run database migrations
alembic upgrade head

# Start the server
uvicorn main:app --reload
```

## API Overview

The platform exposes a RESTful API with JWT-secured endpoints for:

- User authentication and business onboarding
- Sales data upload and management
- Forecast generation and retrieval
- Model performance metrics and comparison
- Inventory recommendation queries

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
