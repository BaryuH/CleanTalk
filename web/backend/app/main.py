from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.cleantalk import router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME)
    origins = [
        "http://localhost:5500",       
        "http://127.0.0.1:5500",
        "http://localhost:5173",       
        "http://localhost:3000",       
        "http://127.0.0.1:3000",
        "null"                         
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"message": "CleanTalk backend running"}

    app.include_router(
        router,
        prefix=settings.API_V1_PREFIX,
        tags=["toxicity"],
    )
    return app
app = create_app()
