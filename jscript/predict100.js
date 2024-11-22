// Esperar a que el documento esté completamente cargado
document.addEventListener('DOMContentLoaded', async function () {
    try {
        // Verificar si TensorFlow.js está cargado
        if (typeof tf === 'undefined') {
            console.error('TensorFlow.js no está cargado. Intentando cargarlo dinámicamente...');
            await loadTensorFlowJs();
        } else {
            console.log("TensorFlow.js ya está cargado.");
            await initializeApp();
        }
    } catch (error) {
        console.error('Error al cargar TensorFlow.js:', error);
        alert('Error al cargar TensorFlow.js. Intenta nuevamente.');
    }
});

// Función para cargar TensorFlow.js si no está cargado
async function loadTensorFlowJs() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs";
        script.onload = function () {
            console.log('TensorFlow.js cargado dinámicamente.');
            resolve();
        };
        script.onerror = function (error) {
            console.error("Error al cargar TensorFlow.js:", error);
            reject(new Error('No se pudo cargar TensorFlow.js.'));
        };
        document.head.appendChild(script);
    });
}

async function initializeApp() {
    console.log("Inicializando la aplicación...");

    // Mensaje de carga
    document.getElementById('loading-message').style.display = "block";

    // Cargar el modelo
    await loadModel();
}

// Cargar el modelo desde la ruta especificada
async function loadModel() {
    let model;
    try {
        console.log('Cargando modelo...');
        const modelPath = './model_kerasnative_v4/model.json'; // Ajusta esta ruta si es necesario
        model = await tf.loadGraphModel(modelPath);
        console.log('Modelo cargado correctamente');
        
        // Actualizar el estado del modelo cargado
        document.getElementById('loading-message').innerText = "Modelo cargado. ¡Listo para hacer predicciones!";
        document.getElementById('loading-message').style.color = "green";  // Mensaje exitoso
        document.getElementById('loading-message').style.display = "block";  // Mostrar mensaje

    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        document.getElementById('loading-message').innerText = `Error al cargar el modelo: ${error.message}`;
        document.getElementById('loading-message').style.color = "red"; // Mostrar mensaje de error
        return;
    }

    // Lógica de predicción cuando el usuario hace clic en el botón
    document.getElementById("predict-button").addEventListener("click", async function () {
        if (!model) {
            alert("El modelo no ha sido cargado aún. Por favor, espera un momento.");
            return;
        }

        try {
            let imageElement = document.getElementById('selected-image');
            let image = tf.browser.fromPixels(imageElement)
                .resizeNearestNeighbor([224, 224])
                .toFloat();
            
            let offset = tf.scalar(127.5);
            image = image.sub(offset).div(offset).expandDims();

            let predictions = await model.predict(image).data();
            console.log("Predicciones:", predictions);
            
            let top5 = Array.from(predictions)
                .map(function (p, i) {
                    return {
                        probability: p,
                        className: TARGET_CLASSES[i] // Asegúrate de que TARGET_CLASSES esté definido
                    };
                })
                .sort(function (a, b) {
                    return b.probability - a.probability;
                })
                .slice(0, 5);
            
            // Mostrar las predicciones
            const predictionList = document.getElementById('prediction-list');
            predictionList.innerHTML = '';  // Limpiar resultados anteriores
            top5.forEach(function (p) {
                predictionList.innerHTML += `<li>${p.className}: ${p.probability.toFixed(6)}</li>`;
            });
            
        } catch (error) {
            console.error('Error durante la predicción:', error);
            document.getElementById('prediction-list').innerHTML = `<li>Error durante la predicción: ${error.message}</li>`;
        }
    });

    // Cargar la imagen seleccionada por el usuario
    document.getElementById("image-selector").addEventListener("change", function () {
        let reader = new FileReader();
        reader.onload = function () {
            document.getElementById("selected-image").src = reader.result;
            document.getElementById('prediction-list').innerHTML = ''; // Limpiar predicciones previas
        };
        reader.readAsDataURL(this.files[0]);
    });
}
