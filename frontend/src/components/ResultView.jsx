const CLASS_COLORS = {
  Animal: '#ff0000',
  Waste: '#00ff00',
  Person: '#0000ff',
};

export default function ResultView({ result }) {
  if (!result) return null;

  const { overlay_image, stats, model } = result;

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = overlay_image;
    link.download = `${model}_prediction.png`;
    link.click();
  };

  return (
    <div className="result-view">
      <h2>Result ({model === 'yolo' ? 'YOLO11s-seg' : 'UNet (ResNet34)'})</h2>

      <img src={overlay_image} alt="prediction overlay" className="result-image" />

      <button onClick={handleDownload}>Download result</button>

      <table className="stats-table">
        <thead>
          <tr>
            <th></th>
            <th>Class</th>
            <th>Pixel %</th>
            {model === 'yolo' && <th>Instances</th>}
          </tr>
        </thead>
        <tbody>
          {Object.entries(stats).map(([className, data]) => (
            <tr key={className}>
              <td>
                <span
                  className="color-swatch"
                  style={{ backgroundColor: CLASS_COLORS[className] }}
                />
              </td>
              <td>{className}</td>
              <td>{data.pixel_percent}%</td>
              {model === 'yolo' && <td>{data.instance_count}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
