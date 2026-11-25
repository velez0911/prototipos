# VisionLine

## Ventajas

- Inspecciona piezas desde una webcam o una camara IP/RTSP.
- Guarda cada deteccion con fecha, lote, pieza y foto.
- Muestra los resultados en una pagina web con registros y graficas.
- Funciona localmente; las imagenes no se envian a un servicio externo.
- Permite marcar como **REVISAR** los casos inciertos.

## Desventajas

- Necesita fotos etiquetadas de piezas buenas y dañadas para entrenar el modelo.
- La precision depende de la calidad y variedad de esas fotos, la iluminacion y la posicion de la camara.
- Detecta piezas por sus colores y registra cada una una sola vez mientras permanece en la imagen. Las capturas de referencia cada 10 segundos se guardan aparte y no aumentan el conteo.
- No controla maquinaria ni reemplaza una inspeccion o sistema de seguridad industrial.

## Como usarlo

### 1. Instalar dependencias (una sola vez)

Abre PowerShell en la carpeta del proyecto. Si ya tienes Python instalado en la ruta habitual de Python 3.12:

```powershell
$python = "$env:LocalAppData\Programs\Python\Python312\python.exe"
& $python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si Python no esta instalado, instala Python 3.12 y vuelve a abrir PowerShell. No pegues el texto `>>`; ejecuta cada comando en una linea separada.

### 2. Preparar fotos y entrenar

Crea estas carpetas y agrega fotos propias de las piezas. El nombre `danada` va sin tilde porque asi lo espera el programa:

```powershell
New-Item -ItemType Directory -Force -Path ".\datos\buena", ".\datos\danada"
```

```text
datos\
  buena\
  danada\
```

Guarda al menos dos imagenes legibles en cada carpeta; se recomiendan muchas mas y en condiciones parecidas a las de la camara de inspeccion. Entrena el modelo:

```powershell
.\.venv\Scripts\python.exe main.py train --data datos --model modelo.joblib
```

Tambien puedes marcar una foto guardada desde el detalle de su lote en el panel: usa **Buena** o **Mala** para actualizar el resultado de esa pieza y copiarla a `datos\buena` o `datos\danada`. Las piezas marcadas manualmente conservan esa clasificación en el registro; las nuevas se clasifican por color y usan el modelo como respaldo cuando el color no es claro.

Desde el mismo detalle puedes usar **Borrar** para eliminar permanentemente el registro y la foto de produccion de esa pieza. El panel pide confirmacion; la copia que ya se haya etiquetado para entrenamiento se conserva.

### 3. Iniciar la inspeccion y abrir el panel

Conecta la webcam y ejecuta:

```powershell
.\.venv\Scripts\python.exe server.py --source 0 --model modelo.joblib
```

En esta computadora, la webcam USB Logi C270 usa `--source 1`. En Windows, VisionLine usa DirectShow para abrir cámaras seleccionadas por índice.
Ejecuta solo una instancia de VisionLine a la vez; el servidor reserva el puerto antes de abrir la cámara y se niega a iniciar si ya está ocupado.
La clasificacion automatica tambien usa el color del lapiz: verde/azul = defectuosa y amarillo/morado = buena. Si no encuentra suficiente color o detecta colores mezclados, usa el modelo entrenado y, si no hay confianza, muestra **REVISAR**.
Antes de clasificar por color, VisionLine busca fragmentos grandes de la pieza separados por una interrupcion visible; si encuentra dos o mas, la marca como **Defectuosa**, incluso si el color normalmente se considera bueno. La deteccion requiere que la fractura tenga suficiente contraste y tamano en la imagen.
Los registros defectuosos guardan el motivo detectado: **Grieta visible**, **Fragmentos separados** o **Color defectuoso**. El panel desglosa estos motivos; los registros antiguos o sin una causa específica aparecen en **Otros**.
Guarda una captura de referencia al iniciar y luego cada 10 segundos en `produccion\capturas`; estas fotos no crean registros de piezas. Una inspección se registra cuando se detecta suficiente área de un color de producto en la imagen, confirmada en dos fotogramas consecutivos; el sistema espera a que el color desaparezca durante tres fotogramas antes de quedar listo para contar la siguiente pieza. El aviso **Producción parada** depende de que no se registren piezas nuevas durante 30 segundos. El botón **Encender/Apagar cámara** del panel libera o vuelve a abrir la webcam y pausa o reanuda la captura y el registro. Si la cámara no está disponible al inicio, el panel sigue accesible para volver a intentar encenderla.

Abre `http://127.0.0.1:8080/` en el navegador. Para usar una camara IP, reemplaza `0` por la URL RTSP entre comillas:

```powershell
.\.venv\Scripts\python.exe server.py --source "rtsp://usuario:contrasena@DIRECCION/stream" --model modelo.joblib
```

La URL RTSP tiene esta forma:

```text
rtsp://USUARIO:CONTRASENA@IP_DE_LA_CAMARA:PUERTO/RUTA_DEL_VIDEO
```

- `rtsp://`: indica que se usara el protocolo RTSP.
- `USUARIO` y `CONTRASENA`: credenciales configuradas para ver el video de la camara.
- `IP_DE_LA_CAMARA`: direccion IP de la camara en la red local, por ejemplo `192.168.1.50`.
- `PUERTO`: puerto RTSP de la camara; con frecuencia es `554`, pero puede ser diferente.
- `RUTA_DEL_VIDEO`: ruta exacta del stream, que depende de la marca y la configuracion de la camara. Consultala en sus ajustes o manual.

Ejemplo de formato (datos ficticios; no lo copies sin reemplazar los valores):

```powershell
.\.venv\Scripts\python.exe server.py --source "rtsp://operador:MI_CLAVE@192.168.1.50:554/stream1" --model modelo.joblib
```

Reemplaza usuario, clave, IP, puerto y ruta por los datos reales; conserva las comillas y ejecuta el comando en una sola linea. La computadora y la camara deben poder comunicarse por la red. Si la clave contiene caracteres especiales como `@`, `:`, `/` o `#`, codificalos para URL (por ejemplo, `@` se escribe `%40`). No compartas ni publiques la URL completa: contiene credenciales de la camara.

El panel muestra las piezas detectadas y sus lotes; los registros y las fotos se guardan en `produccion\`. Detén el servidor con **Ctrl+C** en PowerShell. Si no existe un modelo entrenado, las detecciones se muestran como **REVISAR**.


.\.venv\Scripts\python.exe server.py --source 1 --model modelo.joblib --host 127.0.0.1 --port 8080


 .\.venv\Scripts\python.exe server.py --source 1 --model modelo.joblib