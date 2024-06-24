# weather_prediction
Ejecutar `docker-compose up --build` para crear el contenedor y las imágenes.

*nota:* esperar a que todos las imágenes estén levantadas correctamente antes de iniciar el front. 

 - En `localhost:3000` se encuentra el front-end desde donde se interactúa con la APP.
 - En `localhost:8080` Airflow - donde se puede ingresar a ver los DAG's. User: airflow, pass: airflow 
 - En `localhost:9000` se encuentra minio S3
 - En `localhost:8000` se encuentra el backend y la API.
 - Mediante el puerto 5432, user y pass, se puede conectar a la BD Postgres

*NOTA IMPORTANTE:* La ejecución de los DAGs y el entrenamiento del modelo en background tardan un poco, por lo cual se grabaron videos para mostrar como funcionan realmente. Al iniciar la APP solo van a poder predecir con los modelos ya preentrenados y cargados. (Si esperan lo suficiente ya sería posible ver la creación de los DAGs etc ingresando a airflow, como así el modelo guardado luego de ser entrenado en minio)

- Se trabajó con versionados en DVC y Firebase como storage, con posibilidad de usarlo de hosting. Esto fue porque se iba a subir a producción, pero nos topamos con muchos problemas de configuración inclusive con GCP donde queríamos correr las imágenes, en teoría éste resuelve los protocolos "fácilmente", como Heroku, pero todos pedían medios de pago. 
- La idea de disponibilizarlo era para que no tengan que esperar los entrenamientos y que ya tengan todo cargado en la base de datos.

NOTA: Levantar el front y el back solo si se tienen instaladas las dependencias correspondientes, tanto para python como para react (node)

- Para ejecutar localmente el back y el front, stoppear ammbos contenedores `weather_prediction_dlops (backend)` y `weather_prediction_front`. 
- Ejecutarlo localmente el back moviendose a la carpeta backend `cd backend`, ejecutando el comando `uvicorn  app.main:app --reload`. 
- Para levantar el front moverse a la carpeta frontend `cd frontend` y ejecutar `npm start`.
- Para ejecutar manualmente los DAGS, ejecutar `create_and_load_tables_postgres_train_model`. El DAG `train_random_forest_to_minio` (`train_model_dag.py`) en verdad debería ser el que entrenase, pero por problemas con la libreria que toma las tasks externas para que dependa de otra tarea de otro dag, no logramos que funcione un 100%, por momentos andaba y por otros no, por lo que incluimos todo el funcionamiento en un mismo DAG. 

- Non-issues: 
    - Pueden existir errores en el backend que indiquen que se está intentando acceder al modelo `best_random_forest_model.onnx` que todavía no está guardado en el bucket, la API checkea que esté el modelo para poder ser usado sino utiliza el default. Luego, también se checkea que las tablas `rf_metrics` y `weather_data` estén creadas para poder mostrar las métricas, con lo cual si todavía no hay data (post modelo entrenado mediante el DAG), no se mostrarán tampoco las métricas en el frontend. 
    - Por alguna razón la cuál no hubo tiempo de indagar, el último día de entrega empezo a fallar la inicialización de airflow y por lo tanto la creación de los DAGs ni bien inicia la APP, aún habiendo configurado las variables de entorno para que esto suceda de esta forma. Extrañamente después de unos largos minutos comenzó a funcionar, y a correr correctamente. 
    - Debido a este problema, y los problemas en gran parte de asincronicidad, se tuvo que dejar comentada la parte en que se toma el modelo desde el bucket de minio en el endpoint `/predict/{model_type}` (para el modelo random_forest), ya que al no encontrarlo (porque todavía se está entrenando), el backend quedaba arrojando excepciones y no continuaba su flujo, aún catcheando el error y las excepciones, con la intención de que tome al modelo por default que está en la carpeta `models`. Es un modelo de los que se entrenó previamente en el DAG. 
    - El modelo entrenado y convertido a onnx en el dag, es el que se está utilizando pero de manera particionada, ya que es bastante pesado y github no permitía la subida de archivos de más de 100mb. 

Videos de la aplicación funcional: 
https://drive.google.com/drive/folders/1qu9oYJd9KZX4n8--qyJf95fVBEv7DUE_?usp=drive_link 
