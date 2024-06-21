#!/bin/bash

# Ubicación del script
SCRIPT=$(readlink -f "$0")
# Directorio del script
SCRIPTPATH=$(dirname "$SCRIPT")

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
