# Utilizamos la imagen base de Apache Airflow versión 2.9.2
FROM apache/airflow:2.9.2

# Cambiamos a usuario root para instalar paquetes adicionales
USER root

# Instalamos las dependencias necesarias para compilar ciertos paquetes
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# # Copiamos el archivo init_airflow.sh al contenedor
# COPY init_airflow.sh /init_airflow.sh

# # Cambiamos los permisos del script init_airflow.sh
# RUN chmod +x /init_airflow.sh
# Cambiamos de vuelta al usuario airflow para instalar los paquetes Python
USER airflow

# Copiamos el archivo requirements.txt al contenedor
COPY requirements.txt /requirements.txt

# Actualizamos pip y luego instalamos las dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# # Configuramos el punto de entrada para ejecutar el script de inicialización
# ENTRYPOINT ["/init_airflow.sh"]

