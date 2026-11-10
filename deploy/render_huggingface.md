# Alternativas Gratuitas de Despliegue Backend

Si no deseas configurar una máquina virtual de Oracle Cloud, estas son las dos mejores alternativas gratuitas con soporte HTTPS directo:

---

## Opción 1: Hugging Face Spaces (Docker - Recomendado por Memoria)

Hugging Face Spaces ofrece contenedores gratuitos con **16 GB de RAM** y **2 vCPU**, lo que elimina cualquier riesgo de falta de memoria (*OOM*) con TensorFlow.

### Pasos:
1. Crear una cuenta en [Hugging Face](https://huggingface.co/).
2. Ir a **New Space**:
   - **Name:** `vectoria-api`
   - **License:** MIT o Apache 2.0
   - **Space SDK:** Seleccionar **Docker** > **Blank**.
   - **Space hardware:** Free (CPU basic · 2 vCPU · 16 GB RAM).
3. Conectar el repositorio de GitHub `DiegoMonteroCRC/VectorIA` o subir los archivos del backend (`Dockerfile`, `requirements-backend.txt`, `api/`, `models/`, `data/processed/`, `.keras/`, `src/`).
4. Hugging Face compilará el Dockerfile automáticamente y expondrá la API con HTTPS público permanente:
   ```
   https://<tu-usuario>-vectoria-api.hf.space
   ```
5. Endpoint de prueba:
   ```
   https://<tu-usuario>-vectoria-api.hf.space/docs
   ```

---

## Opción 2: Render (Web Service Free)

Render permite desplegar el Dockerfile directamente conectando tu cuenta de GitHub.

### Pasos:
1. Iniciar sesión en [Render.com](https://render.com/).
2. Crear un **New Web Service**.
3. Conectar el repositorio `DiegoMonteroCRC/VectorIA`.
4. Configuración:
   - **Environment:** `Docker`
   - **Branch:** `feature/preparacion-despliegue-produccion` (o `main` tras merge)
   - **Dockerfile Path:** `./Dockerfile`
   - **Plan Type:** `Free` (0.1 CPU, 512 MB RAM)
5. En **Environment Variables**, añadir:
   - `PORT=8000`
   - `TF_ENABLE_ONEDNN_OPTS=0`
   - `TF_CPP_MIN_LOG_LEVEL=2`
6. Render proporcionará una URL pública HTTPS:
   ```
   https://vectoria-api.onrender.com
   ```
*(Nota sobre Render Free: se suspende tras 15 minutos de inactividad; la primera petición tarda ~50 segundos en despertar).*
