#!/bin/bash
set -e

# Upgrade de la base de datos
airflow db migrate

# Crear usuario admin si no existe
airflow users create \
    --username airflow \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password airflow \
    || true  # Permite que el script continúe si el usuario ya existe

# Crear roles (sin descripción en Airflow 2.9.2)
airflow roles create viewer
airflow roles create editor

echo "Configuración inicial de usuario de Airflow completada exitosamente."
