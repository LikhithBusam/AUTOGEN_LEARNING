#!/usr/bin/env python3
"""
Minimal test server to check if FastAPI is working
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Test server is working!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting minimal test server...")
    print("🌐 Open http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
