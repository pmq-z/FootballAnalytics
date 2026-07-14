# Predictor de partidos | K-Means + ID3
Creado a partir de un examen que buscaba predecir los resultados de **España vs Francia (14/07/2026)** y fue modificado para cualquier par de selecciones del roster.
Agradezco a Diego de Luna (Coworker) por su opinión, observaciones y recomendaciones durante el desarrollo de este proyecto.

## El veredicto

**La predicción, antes del partido**
<img width="1300" height="936" alt="Prediccion" src="https://github.com/user-attachments/assets/e16c6ced-8ebd-4670-944d-c208543e247d" />

**El resultado real, después del partido**
<img width="676" height="492" alt="Resultado" src="https://github.com/user-attachments/assets/75a2e2b4-ce08-48ac-8567-a6b708a8758c" />

Ganador correcto. Goles de España, clavados. Se le fue el gol de Francia — el modelo le concedía a Mbappé la probabilidad de anotador más alta del partido (47.9%) y el 0 del marcador dice que no cayó.

Un fallo de un gol en el visitante y un acierto pleno en ganador y marcador local. Para un árbol entrenado sobre clusters de estilo de juego, no está mal.

## Arranque
```bash
pip install -r requirements.txt
python app.py
```
`localhost:5000`, eleccion de equipos, apretar el botón rojo. Listo.

## Qué hace?
1. **Jala datos:** Últimos 12 partidos de cada equipo, el historial head-to-head y las plantillas con minutos/goles (`data_source.py`).
2. **K-Means:** Normaliza con StandardScaler y agrupa los partidos por *cómo se jugaron* (goles totales, xG, tiros, desbalance de posesión). Nunca ve quién ganó. Salen perfiles tipo *partido cerrado*, *partido abierto*, *dominio unilateral*, *duelo táctico*.
3. **Inyección:** el número de cluster se convierte en feature categórica.
4. **ID3:** un árbol de decisión con criterio `entropy` toma ese cluster + contexto (elo, forma, dominio en el H2H) y escupe el ganador. La interfaz te muestra la ruta exacta que siguió el árbol, nodo por nodo.
5. **Marcador y anotadores:** Poisson restringido al veredicto del árbol, y los goleadores se ponderan por goles/90 × minutos × forma.

También hay un botón para descargar el dataset en `.xlsx` con los clusters ya asignados.

## Archivos
```
app.py                   - servidor Flask
data_source.py           - conector de datos (mocks con la forma de API-Football)
model_pipeline.py        - K-Means + ID3 + Poisson
templates/index.html     - la vista
static/css/styles.css    - neo-brutalismo, paleta New Caloric (blanco/negro/#FF0033)
static/js/main.js        - interacciones
docs/                    - capturas
```
Los datos son mocks deterministas. Para producción solo hay que cambiar el cuerpo de las funciones que traen partidos, head-to-head y plantillas; el resto del pipeline no se entera.

> Nota para quien lea el código: la nomenclatura interna de variables usa referencias a Monster Hunter. Fue puro capricho mío.

## V1 (Deprecated)
La primera versión del algoritmo tomaba de una API los últimos partidos de Francia y España, junto con el historial de partidos históricos Francia vs España (Eurocopas, Mundial, Nations League, clasificatorios, 1984-2025).
Utilicé **K-Means clustering normalizado** (StandardScaler + K-Means) sin contar con la etiqueta de "ganador" como feature del clustering, puesto que esta se computa **después** del cluster; con esto me refiero a que "ganador" no fue trabajada como label, sino como una fórmula (`dif_goles > 0 → España`, etc.). El algoritmo solo agrupa partidos según sus goles/diferencia/total, y el nombre del ganador se calcula aparte, para lectura humana, no como insumo del modelo.
Con esos clusters + un modelo de goles esperados (Poisson) + simulación Monte Carlo, V1 ya entregaba: dataset, análisis y conclusión (Ganador / Marcador / Anotadores).

## V2 (Actual)
Lo que cambió: el ganador ya no sale solo de la fórmula, ahora entra un **árbol ID3** que aprende a decidir usando el cluster como input. El Poisson sigue ahí, pero degradado a calcular el marcador *dentro* de lo que el árbol ya dictaminó. Además: interfaz, export a Excel y soporte para más equipos.

## Aviso
Esto es un juguete estadístico. Acertó una vez. Eso no lo convierte en un oráculo, no apuestes... pero, y si sí?
