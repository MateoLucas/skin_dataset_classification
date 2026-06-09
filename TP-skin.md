**1. Dataset y Preprocesamiento**



¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?





Porque las capas internas de la red tienen una matriz de pesos fija, que se entrena con entradas de un tamaño particular. Supongamos que definimos una cantidad de neuronas de entrada de 60 000. Si la imagen es muy grande, digamos 90 000 neuronas, entonces truncaría, dejaría 30 000 píxeles afuera, y se pierde la relación geométrica. La imagen deja de tener sentido, y se entrena el modelo sobre un dato que no es representativo de la realdiad. En camio si se entra con 30 000 píxeles, quedarías otros 30 000 en negro, como si la piel de l imagen fuese negra, que tampoco es un dato representativo de la realidad. Las imágenes caradas no tendrían sentido.

También podría verse desde otro ángulo. Si la matriz de pesos tiene tamaño fijo, algebraicamente no se puede multiplicar por vectores de tamaños distintos.



¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?



Porque es una librería muy optimizada con C++, lo que lo hace más rápida.



¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?



Los valores de los píxeles orginalmente van de 0 a 255. Al normalizarlos, se les resta un valor medio y se los divide por su desviación estándar. Esto escala los valores a un rango chico y centrado. Supongamos que tomamos en cuenta un píxel de 255 y uno de 10, y que los pesos iniciales son ambos 0,1. Entonces z=(0,1.255)+(0,1.10)+b=26,5. Acá el píxel con valor 10 pierde influencia porque la matemática le quita preponderancia, pero no es menos significativo desde un punto de vista de información.

Además, el optimizador actualiza los pesos en función de la gradiente. Si los saltos de neurona a neurona son muy bruscos, los cambios son muy radicales, y la red no puede aprender correctamente.



¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?



Transforma la estructura del touch.Tensor HWC (Altura, Ancho, Canales) a uno que PyTorch entienda, CHW (Canales, altura, ancho)





**2. Arquitectura del Modelo**



¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

La usamos como punto de partida para comparar la CNN. La MLP no tiene una estructura que nativamente entienda imágenes de 2 dimensiones. La CNN se diseñó especialmente para analizar imágenes bidimensionales. La tecnología sólo tiene sentido si es superadora de una MLP.



¿Qué hace la capa `Flatten()` al principio de la red?

&nbsp;Pone en fila todos los datos de la imagen. Lee primero todo el canal rojo, luego el verde, y luego el azul. Y cada conjunto de píxeles por canal lo lee como un libro. Esto destruye la referencia espacial que tienen los píxeles. Una vez aplanada, se pierde la referencia de qué píxel estaba encima de cuál otro. Sería en cierta forma como entrenar una red para identificar caballos pero en cada foto la cabeza del caballo está en un lugar distinto, la pierna, la cola, en la montaña del fondo en el piso. No estaría entrenando la red para ver un caballo como corresponde.

&nbsp;



¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

se usó ReLU=max(0,x) para que si el valor de la neurona es negativo (insisundo que la relación de esa neurona con el resultado final es bajo) se compute como 0 directamente. Esto simplifica el cálculo (0.x=0) y agrega alinealides. También, mantiene proporcional el crecimiento con x cuando es positivo, no ocurre el Vanishing Gradient que ocurre con Tanh y Sigmoid. Esto es que cuando crece mucho x, el valor del gradiente tiende a 0 y no se mueve, y por ende deja de aprender.





¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?-

&nbsp;Debe modificarse el valor de `in\_features` de la primera capa `nn.Linear`, recalculándolo con la nueva fórmula: nuevoAlto, nuevoAncho, Canales.



**3. Entrenamiento y Optimización**



¿Qué hace `optimizer.zero\_grad()`?

&nbsp;Reinicia a cero los gradientes almacenados de todas las neuronas del modelo. Por diseño, PyTorch acumula los gradientes en cada llamada a .backward(). Si no se ejecutase esta línea, el optimizador mezclará los errores del batch anterior con el batch actual, generando que los pesos se actualicen de manera errónea.



¿Por qué usamos `CrossEntropyLoss()` en este caso?

&nbsp;Es la función estándar para clasificación multiclase excluyente. Utiliza LogSoftmax, que obiene de salida de la red un conjunto de probabilidades que suman 1, con la pérdida de probabilidad logarítmica negativa NLLLoss, que penaliza exponencialmente al modelo si se equivoca con seguridad.



¿Cómo afecta la elección del tamaño de batch (`batch\_size`) al entrenamiento?

&nbsp;Un batch chico ayuda a escapar de mínimos locales. Si se calcula el gradiente de un dataset entero, el optimizador "caminará" en esa dirección hasta llegar a gradiente 0. Pero eso puede ser un mínimo local. En cambio si recalcula gradiente paso por paso, cuando se encuentre en un mínimo local, para el siguiente batch ese mismo punto no va a ser un mínimo local, y se va a mover el optimizador.



¿Qué pasaría si no usamos `model.eval()` durante la validación?

&nbsp;Dropout apaga neuronas al azar durante el entrenamiento para generar un modelo más robusto. Si no usamos model.eval() Dropout seguirá apagando neuronas en el momento en el que queremos evaluar al modelo, poniendole un palo cuando queremos asesorar su capacidad, y no sólo en su entrenamiento. Además, hace que BatchNorm deje de calcular estadísticas de solamente las imágenes de validación, y no del promedio histórico.



**4. Validación y Evaluación**



¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Que el modelo hizo Overfitting. Memorizó el set de entrnamiento, pero cuando se le presentaron imágenes nuevas, le costó entenderlas. Significa que la calidad del aprendizaje se ve comprometida.



¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

&nbsp;En el contexto de la medicina, los falsos negativos son mucho más problemáticos que los falsos positivos. El Recall/Sensibilidad mide justamente eso.

Miarar la precisión nos dará una idea de cuántos falsos positivos tendremos.

El F1-Score balancea estas dos métricas.



¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

&nbsp;La matríz de confusión nos deja ver con que accuracy la red detecta cada enfermedad puntual, cuál enfermedad se confunde más con cual, y si hay alguna en particular que tenga problemas.





**5. TensorBoard y Logging**



¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

&nbsp;Te permite ver la evolución del entrenamiento "en vivo" mediante gráficas visuales dinámicas. Puedes detectar si el modelo se estancó, si empezó a hacer overfitting, e inspeccionar visualmente imágenes y reportes sin necesidad de saturar la consola de texto de tu terminal.



¿Qué diferencias hay entre loguear `add\_scalar`, `add\_image` y `add\_text`?

add\_scalar: Guarda un único número flotante asociado a una época (ej. curvas 2D de Loss o Accuracy).

add\_image: Envía tensores tridimensionales para renderizar imágenes reales en la interfaz.

add\_text: Registra cadenas de caracteres complejas (útil para imprimir reportes de clasificación en tablas formateadas con Markdown o HTML).





¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

&nbsp;Te permite ver que el cargador de datos no esté rompiendo la imagen, verificar que los reescalados y normalizaciones mantengan la legibilidad y analizar qué características físicas tienen las imágenes donde el modelo está fallando el examen.



¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Se guardan los archivos de desempeño en carpetas separadas para poder graficar por separado a posteriori. 



**6. Generalización y Transferencia**



¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

&nbsp;Solo se debería cambiar el parámetro out\_features de la última capa lineal de la red para que devuelva un tamaño de salida igual a 100.



¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

&nbsp;Porque las CNNs mueven un kernel a lo largo de la imágen bidimensional y extrae features de ese barrido. Entonces mira los píxeles en el contexto adecuado de la imágen.



¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase? La red memoriza esos pocos ejemplos específicos (overfitting), perdiendo la capacidad de entender si hay variaciones de luz, ángulos o fondos de nuevas imágenes.



¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Mirando solamente uno de los 3 canales. Por ejemplo el Red. 



**7. Regularización (Respuestas Teóricas)**



¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

&nbsp;Es un conjunto de técnicas para evitar el overfitting, como Dropout, Data augmentation, monitorear la curva de Loss de validación a lo largo de las épocas, o regularización L1 y L2. Regularización L1 pone un 0 en los pesos de las neuronas que no aportan mucho. Regularización L2 penaliza con el cuadrado del valor de los pesos, para que la red no aprenda detalles puntuales y las neuronas tengan valores mejor distribuidos.



¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Dropout apaga de forma aleatoria un porcentaje de neuronas en cada pasada, para eliminar codependencias entre ellas y obligándolas a aprender características independientes.

L2 (Weight Decay) modifica la función de pérdida para que castigue los pesos inusualmente grandes. Esto hace que la red no busque recordar detalles demasiado puntales que sean específicos del dataset de entrenamiento. 



¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

Es una capa que normaliza las salidas de la capa anterior dentro de cada "mini" batch, forzando a que la media sea cero y la varianza sea uno. Ayuda a mitigar el Internal Covariate Shift (cambios bruscos en la distribución de datos que reciben las capas profundas cuando cambian los pesos de las primeras capas), estabilizando el flujo del entrenamiento.



¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

&nbsp;Al mantener las escalas/dimensiones de las activaciones controladas en toda la red, te permite configurar learning rates significativamente más altas sin el riesgo de que los gradientes exploten. Esto hace que el modelo avance a pasos grandes y converja en muchas menos épocas.



¿Can `BatchNorm` actuar como regularizador? ¿Por qué?

Sí, actúa como un regularizador suave. Dado que calcula la media y la varianza basándose únicamente en el lote actual (el cual es una muestra aleatoria del dataset), introduce un pequeño ruido estadístico en las activaciones de las neuronas. Este ruido constante previene que la red se ajuste demasiado al ruido de los datos, produciendo un efecto similar al del Dropout.



¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

&nbsp;La curva de Train Loss desciende perfectamente hacia cero, mientras que la curva de Val Loss deja de bajar y empieza a curvarse hacia arriba (efecto rebote). También, la brecha (gap) entre el gráfico de Train Accuracy y Val Accuracy se vuelve cada vez más grande.



¿Cómo ayuda la regularización a mejorar la generalización del modelo?

&nbsp;Al imponer restricciones (borrando neuronas con Dropout, achicando pesos con L2 o distorsionando datos con Data Augmentation), le prohíbe al modelo aprender los detalles irrelevantes de las imágenes de entrenamiento. La red se ve obligada a aprender solo los patrones más genéricos.



**8. Inicialización de Parámetros (Respuestas Teóricas)**



¿Por qué es importante la inicialización de los pesos en una red neuronal?

&nbsp;Porque determina el punto de partida del explorador en el mapa de optimización. Una mala inicialización inicial puede hacer que las señales se desvanezcan (lleguen a cero) o exploten (se vuelvan infinitas) en las primeras capas antes de que la red tenga oportunidad de dar su primer paso de aprendizaje.



¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

&nbsp;Ocurre el problema de la simetría. Si todas las neuronas de una capa oculta inician con los mismos pesos, calcularán exactamente la misma salida en el forward pass y recibirán exactamente el mismo gradiente en el backward pass. Como consecuencia, todas se actualizarán de forma idéntica, haciendo que la red actúe como si tuviera una sola neurona por capa, perdiendo toda su capacidad de aprender.



¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

Xavier: Está calibrada matemáticamente asumiendo que la red utiliza activaciones lineales o de tipo espejo como Tanh o Sigmoid.



He (Kaiming): Está formulada específicamente para ReLU. Como ReLU anula de golpe la mitad del espacio de entrada (los valores negativos), He duplica matemáticamente la varianza de los pesos para compensar la pérdida de información provocada por las neuronas que ReLU "apaga", y previene Vanishing Gradient.





¿Por qué en una red con ReLU suele usarse la inicialización de He?

Aumentar la intensidad de la señal y evitar el Vanishing Gradient que ReLU genera intrínsicamente por apagar neuronas con valores negativos.



¿Qué capas de una red requieren inicialización explícita y cuáles no?

Solo necesitan iniciaiación las capas entrenables internas como matrices de pesos y vectores de sesgo. Capas como `nn.Flatten`, `nn.ReLU`, `nn.MaxPool2d` o `nn.Dropout` no contienen parámetros propios y no requieren inicialización.



¿Por qué `bias` se suele inicializar en cero?

Inicializar los pesos es necesario para que el sistema tenga un punto de partida que optimizar. Inicializar los sesgos con un valor distinto de cero es innecesario, porque inicializar los pesos es suficiente, e incluso podrían causar saturación.

