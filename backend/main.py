from contextlib import asynccontextmanager
from pathlib import Path

from backend.api import (
    automl,
    dataset_acquisition,
    datasets,
    experiments,
    imports,
    ml,
    models,
    projects,
    runs,
    sessions,
)
from backend.config import settings
from backend.database import close_db, init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(sessions.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(models.router)
app.include_router(ml.router)
app.include_router(dataset_acquisition.router)
app.include_router(automl.router)
app.include_router(runs.router)
app.include_router(imports.router)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}


@app.get("/")
async def root():
    return {"message": "ML Assistant API", "version": settings.app_version, "docs": "/docs"}
