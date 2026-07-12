import { useState } from 'react';
import UploadForm from './components/UploadForm.jsx';
import ResultView from './components/ResultView.jsx';
import { predictImage } from './api.js';

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (file, model) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImage(file, model);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Garbage Segmentation Demo</h1>
      <p>Upload an image and choose a model to see predicted Waste / Animal / Person segmentation.</p>

      <UploadForm onSubmit={handleSubmit} loading={loading} />

      {error && <p className="error-message">Error: {error}</p>}

      <ResultView result={result} />
    </div>
  );
}
