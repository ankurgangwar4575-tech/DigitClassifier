import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import "./App.css";

type ModelName = "CNN" | "LSTM" | "RNN";

type Prediction = {
  digit: number;
  model: string;
  confidence: number;
  inference_time_ms: number;
  test_accuracy: number;
};

const MODELS: ModelName[] = ["RNN", "LSTM", "CNN"];
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function App() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("CNN");
  const [image, setImage] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setImage(event.target.files?.[0] ?? null);
    setPrediction(null);
    setMessage("");
  }

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!image) {
      setMessage("Please choose a digit image first.");
      return;
    }

    setIsLoading(true);
    setMessage("");
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append("model", selectedModel);
      formData.append("file", image);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Prediction request failed.");
      setPrediction(await response.json());
    } catch {
      setMessage("Unable to reach the backend. Start it, then try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="terminal-card" aria-labelledby="title">
        <div className="hero-copy">
          <span className="eyebrow">AI IMAGE RECOGNITION</span>
          <h1 id="title">Digit <em>Classifier</em></h1>
          <p>Upload a handwritten number and let your selected neural network identify it.</p>
        </div>

        <form onSubmit={handlePredict}>
          <fieldset>
            <legend>01. Choose a model</legend>
            <div className="model-options">
              {MODELS.map((model) => (
                <label key={model} className="radio-option">
                  <input checked={selectedModel === model} name="model" onChange={() => setSelectedModel(model)} type="radio" value={model} />
                  <span className="model-name">{model}</span>
                  <small>{model === "CNN" ? "Best image accuracy" : model === "LSTM" ? "Sequence learning" : "Fast baseline"}</small>
                </label>
              ))}
            </div>
          </fieldset>

          <div className="upload-section">
            <label className="section-label" htmlFor="digit-image">02. Upload your digit</label>
            <label className="upload-box" htmlFor="digit-image">
              <span className="upload-icon">↑</span>
              <span><strong>{image ? image.name : "Choose an image"}</strong><small>PNG or JPG · a clear handwritten digit works best</small></span>
            </label>
            <input accept="image/png,image/jpeg,image/jpg" id="digit-image" onChange={handleFileChange} type="file" />
          </div>

          <button className="predict-button" disabled={isLoading} type="submit"><span>{isLoading ? "Analysing..." : "Predict digit"}</span><b>→</b></button>
        </form>
        {message && <p className="message" role="alert">{message}</p>}
      </section>

      {prediction && (
        <section className="result-card" aria-live="polite">
          <p className="result-eyebrow">PREDICTION COMPLETE</p>
          <h2>Your digit is <strong>{prediction.digit}</strong></h2>
          <dl>
            <div><dt>Model</dt><dd>{prediction.model}</dd></div>
            <div><dt>Confidence</dt><dd>{prediction.confidence.toFixed(2)}%</dd></div>
            <div><dt>Inference Time</dt><dd>{prediction.inference_time_ms.toFixed(1)} ms</dd></div>
            <div><dt>Test Accuracy</dt><dd>{prediction.test_accuracy.toFixed(2)}%</dd></div>
          </dl>
        </section>
      )}
    </main>
  );
}

export default App;
