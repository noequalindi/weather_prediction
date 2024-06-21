# Usa una imagen de node como base para construir la aplicación
FROM node:alpine 

# Establece el directorio de trabajo
WORKDIR /app

# Copia el package.json y package-lock.json al contenedor de trabajo
COPY package*.json .

# Instala las dependencias
RUN npm ci --only=production

# Copia el resto de los archivos al contenedor de trabajo, carpeta public, imgs etc.
COPY . .

# Construye la aplicación para producción
RUN npm run build

# puerto por convencion
EXPOSE 3000 

# correr el proceso principal del contenedor
CMD ["npm", "start"]
