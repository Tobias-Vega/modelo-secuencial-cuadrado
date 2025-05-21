const fitButton = document.getElementById('fit-btn');
const inputNumber = document.getElementById('input-number');
const predictButton = document.getElementById('predict-btn');
const result = document.getElementById('result');
const predictionDiv = document.getElementById('prediction');
const graficoDiv = document.getElementById('grafico');

let model;

async function fitModel() {
  model = tf.sequential();

  model.add(tf.layers.dense({ units: 8, inputShape: [1], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd',
  });

  const xs_raw = [-6, -5, -4, -3, -2, -1, 0, 1, 2];
  const ys_raw = [91, 66, 45, 28, 15, 6, 1, 0, 3];

  const xs = tf.tensor2d(
    xs_raw.map((x) => x / 10),
    [xs_raw.length, 1],
  );
  const ys = tf.tensor2d(
    ys_raw.map((y) => y / 100),
    [ys_raw.length, 1],
  );

  const surface = {
    name: 'Pérdida durante el entrenamiento',
    tab: 'Entrenamiento',
  };

  const history = await model.fit(xs, ys, {
    epochs: 700,
    callbacks: tfvis.show.fitCallbacks(surface, ['loss'], {
      callbacks: ['onEpochEnd'],
    }),
  });

  const losses = history.history.loss;
  const initialLoss = losses[0].toFixed(4);
  const finalLoss = losses[losses.length - 1].toFixed(4);
  const reduction = ((initialLoss - finalLoss) / initialLoss) * 100;

  graficoDiv.style.display = 'block';

  document.getElementById(
    'initial-loss',
  ).textContent = `Pérdida inicial: ${initialLoss}`;
  document.getElementById(
    'final-loss',
  ).textContent = `Pérdida final: ${finalLoss}`;
  document.getElementById(
    'reduction',
  ).textContent = `Reducción: ${reduction.toFixed(4)}%`;
}

predictButton.addEventListener('click', () => {
  const inputValue = inputNumber.value;

  if (inputValue.trim() === '') {
    alert('Ingrese uno o más números separados por coma');
    return;
  }

  const values = inputValue.split(',').map(Number);

  const scaledValues = values.map(x => x / 10);

  const tensor = tf.tensor2d(scaledValues, [values.length, 1]);

  const prediction = model.predict(tensor);

  prediction.array().then((predict) => {
    result.innerHTML = values
      .map((value, i) => {
        const descaled = predict[i][0] * 100;
        return `El resultaddo de predecir ${value} es: ${descaled.toFixed(
          2,
        )}`;
      })
      .join('<br>');
  });
});

fitButton.addEventListener('click', () => {
  fitModel();

  alert('Entrenamiento del modelo finalizado');

  predictionDiv.style.display = 'block';
});
