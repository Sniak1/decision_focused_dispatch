# Experimento 01 — Single-bus Application-Driven Learning

Este directorio contiene una reproducción **didáctica y transparente** del caso de una barra de:

> J. Dias Garcia, A. Street, T. Homem-de-Mello, F. D. Muñoz, *Application-Driven Learning: A Closed-Loop Prediction and Optimization Approach Applied to Dynamic Reserves and Demand Forecasting*, Operations Research 73(1).

El objetivo de este primer experimento no es reproducir cada número del artículo, sino reproducir el mecanismo que queremos entender antes de pasar a *Forecasting outside the Box* y *Decision Focused Scenario Generation*.

## Qué está tomado directamente del paper

- 1 barra, 1 carga y 4 generadores.
- Capacidades: `[5, 5, 2.5, 2.5]`.
- Costos marginales: `[1, 2, 4, 8]`.
- Costo de load shedding = 8 veces el costo del generador más caro.
- Costo de spillage = 3 veces el costo del generador más caro.
- Cada generador puede asignar hasta 30% de su capacidad como reserva.
- Costo de asignación de reserva = 30% del costo nominal del generador.
- Proceso de demanda AR(1): `D_t = 0.6 + 0.9 D_{t-1} + eps_t`.
- Media de largo plazo 6 y coeficiente de variación 0.4.
- Demandas negativas truncadas a 0.
- Forecast: `D_hat_t = theta0 + theta1 D_{t-1}`.
- Reservas forecast AR(0): constantes `R_up` y `R_down`.
- Benchmark LS-Ex: LS para demanda y reserva exógena igual a `1.96 * sigma_residual` en ambas direcciones.
- Entrenamiento application-driven mediante búsqueda derivative-free tipo Nelder-Mead.

## Decisión técnica que NO queda completamente especificada en el texto del paper

El artículo fija el coeficiente de variación del **proceso de demanda**, pero el texto no entrega directamente la desviación estándar de la innovación `eps_t`. En este código la inferimos imponiendo que el AR(1) gaussiano sin truncar tenga:

`std(D) = 0.4 * 6 = 2.4`.

Para un AR(1) estacionario:

`sigma_D = sigma_eps / sqrt(1 - 0.9^2)`

por lo que usamos:

`sigma_eps = 2.4 * sqrt(1 - 0.9^2) ≈ 1.0461`.

Después se truncan las demandas negativas a cero, como indica el paper. Esto altera levemente los momentos efectivos. Por eso **no debemos exigir que nuestros números coincidan exactamente con las figuras originales** sin revisar el código suplementario de los autores.

## Modelo de planificación

Para cada forecast de demanda `D_hat` y requisitos de reserva `R_up`, `R_down` resolvemos:

```text
min  c'g + p'r_up + p'r_down + lambda_LS*delta_LS + lambda_SP*delta_SP

s.a.
     sum(g) + delta_LS - delta_SP = D_hat
     sum(r_up)   = R_up
     sum(r_down) = R_down
     g + r_up <= K
     g - r_down >= 0
     r_up <= rbar
     r_down <= rbar
     variables >= 0
```

En una barra no hay restricciones de transmisión.

## Evaluación ex-post

Una vez fijados `g*`, `r_up*`, `r_down*`, la generación real puede moverse dentro de:

```text
g* - r_down* <= g_real <= g* + r_up*
```

El costo evaluado es el costo fijo de la decisión planificada más las penalizaciones por déficit o spillage reales. En una barra la segunda etapa se puede evaluar analíticamente, por lo que no resolvemos un segundo LP innecesario.

## Instalar

Con tu `.venv` activado:

```powershell
pip install -r requirements.txt
```

## Primera ejecución recomendada

Desde la raíz del proyecto:

```powershell
python experiments/01_single_bus_application_driven.py
```

Por defecto usa:

- 150 observaciones de entrenamiento;
- 2000 observaciones de test;
- LS-Ex y Opt-Opt;
- 60 iteraciones máximas de Nelder-Mead.

Es una configuración de aprendizaje, no la reproducción estadística completa de las 100 repeticiones del paper.

Para incluir los cuatro modelos del artículo:

```powershell
python experiments/01_single_bus_application_driven.py --all-models
```

Para acercarnos al esquema de evaluación del paper (test de 10 000 observaciones):

```powershell
python experiments/01_single_bus_application_driven.py --train-size 250 --test-size 10000 --maxiter 250 --all-models
```

Esa ejecución puede tardar bastante más porque cada evaluación de Nelder-Mead resuelve un LP por observación de entrenamiento.

## Qué queremos observar

No buscamos principalmente el menor RMSE. Queremos comprobar si un forecast deliberadamente sesgado puede producir un menor costo operativo.

En particular, compara en `results/single_bus/summary.csv`:

- `test_cost`
- `test_rmse`
- `test_mean_error_actual_minus_forecast`
- `reserve_up`
- `reserve_down`

Si `actual - forecast` tiene media negativa, el modelo está **sobreestimando** sistemáticamente la demanda.

El paper reporta justamente que el modelo completamente optimizado tiende a sesgar el forecast hacia arriba debido a la asimetría entre el costo de load shedding y el costo de spillage.

## Advertencia sobre la heurística

Nelder-Mead resuelve un problema no convexo y no garantiza encontrar el óptimo global. El paper compara su heurística contra una formulación exacta en instancias pequeñas; nuestro primer script implementa solo la ruta heurística porque es la que queremos entender y porque no requiere Gurobi/BilevelJuMP. Por eso, una corrida corta puede terminar en un mínimo local y no mostrar inmediatamente el mismo sesgo hacia arriba reportado por los autores. Eso no se debe interpretar como una refutación del paper. Más adelante haremos múltiples semillas/reinicios y, si hace falta, contrastaremos con una búsqueda más exhaustiva en el sistema de una barra.
