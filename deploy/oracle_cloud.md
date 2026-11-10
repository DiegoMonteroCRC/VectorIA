# Guía de Despliegue en Oracle Cloud Always Free (24/7 Gratis)

Oracle Cloud ofrece instancias virtuales gratuitas de por vida (**Always Free**) ideales para mantener el backend de VectorIA activo de forma permanente sin costo y sin apagarse por inactividad.

---

## 1. Crear la Instancia en Oracle Cloud

1. Iniciar sesión en [Oracle Cloud Console](https://cloud.oracle.com/).
2. Ir a **Compute** > **Instances** > **Create Instance**.
3. **Nombre:** `vectoria-backend`.
4. **Imagen y Forma (Image and Shape):**
   - **Imagen:** Ubuntu 22.04 o 24.04 Minimal (Always Free-eligible).
   - **Forma:** 
     - *Opción A (Recomendada):* **Ampere ARM (VM.Standard.A1.Flex)** con 2 a 4 OCPUs y 6 a 12 GB RAM (Always Free).
     - *Opción B:* **AMD (VM.Standard.E2.1.Micro)** con 1 OCPU y 1 GB RAM (Always Free).
5. **Red:** Asignar una IP pública IPv4 automática.
6. **SSH Keys:** Descargar la clave privada (`id_rsa`) para conectarse por SSH.
7. Hacer clic en **Create**.

---

## 2. Abrir los Puertos en Oracle Cloud (Security List)

Por defecto, Oracle bloquea el tráfico entrante salvo el puerto 22.

1. En los detalles de la instancia, hacer clic en la **Subnet**.
2. Hacer clic en **Default Security List for...**.
3. Añadir **Ingress Rule**:
   - **Source CIDR:** `0.0.0.0/0`
   - **IP Protocol:** TCP
   - **Destination Port Range:** `80,443,8000`
   - **Description:** Permitir tráfico HTTP/HTTPS y API.

---

## 3. Conexión e Instalación de Docker en el Servidor

Conectarse desde la terminal o PowerShell:

```bash
ssh -i /ruta/a/tu/clave.key ubuntu@<IP_PUBLICA_ORACLE>
```

Actualizar el sistema e instalar Docker y Git:

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose git
sudo usermod -aG docker $USER
newgrp docker
```

---

## 4. Clonar el Repositorio y Levantar el Servicio

```bash
git clone https://github.com/DiegoMonteroCRC/VectorIA.git
cd VectorIA
git checkout feature/preparacion-despliegue-produccion

# Levantar el contenedor en segundo plano con auto-reinicio
docker-compose up -d --build
```

Verificar que la API esté respondiendo:

```bash
curl http://localhost:8000/health
```

---

## 5. Habilitar HTTPS Público y Gratuito

Hay dos formas sencillas de obtener HTTPS con certificado SSL automático sin costo:

### Opción A: Caddy Server (Reverse Proxy automático con SSL)

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install caddy
```

Editar `/etc/caddy/Caddyfile`:

```caddyfile
# Si tienes un dominio gratuito (ej. de DuckDNS o No-IP):
tu-subdominio.duckdns.org {
    reverse_proxy 127.0.0.1:8000
}
```

Reiniciar Caddy:
```bash
sudo systemctl restart caddy
```

### Opción B: Cloudflare Tunnel (100% Gratis, sin abrir puertos ni configurar DNS)

1. Crear cuenta gratuita en [Cloudflare](https://dash.cloudflare.com/).
2. Ir a **Zero Trust** > **Networks** > **Tunnels** > **Create Tunnel**.
3. Seguir el comando de 1 sola línea para instalar `cloudflared` en el servidor Ubuntu.
4. En la configuración del túnel, apuntar el servicio a:
   - **Type:** `HTTP`
   - **URL:** `localhost:8000`
5. Cloudflare asigna automáticamente una URL pública HTTPS (`https://api-vectoria.tudominio.com`).
