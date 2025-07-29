from celery import Celery
import os

def make_celery():
    return Celery(
        "tasks",
        broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

celery = make_celery()
