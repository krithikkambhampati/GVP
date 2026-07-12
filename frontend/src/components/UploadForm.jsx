import { useState } from 'react';

export default function UploadForm({ onSubmit, loading }) {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState('yolo');
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    if (selected) {
      setPreviewUrl(URL.createObjectURL(selected));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onSubmit(file, model);
  };

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <div className="form-row">
        <label htmlFor="file-input">Upload image</label>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />
      </div>

      <div className="form-row">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="yolo">YOLO11s-seg</option>
          <option value="unet">UNet (ResNet34)</option>
        </select>
      </div>

      {previewUrl && (
        <div className="preview">
          <p>Preview:</p>
          <img src={previewUrl} alt="preview" />
        </div>
      )}

      <button type="submit" disabled={!file || loading}>
        {loading ? 'Running inference...' : 'Predict'}
      </button>
    </form>
  );
}
