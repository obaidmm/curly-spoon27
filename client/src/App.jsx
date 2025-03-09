import { useState } from 'react';
import './App.css';

function App() {
  const [imagePath, setImagePath] = useState(''); // Image path to be processed
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [detectedImage, setDetectedImage] = useState(null); // Processed image
  const [averageConfidence, setAverageConfidence] = useState(null); // Average confidence


  // Update the backend URL for local testing or Raspberry Pi
  const backendUrl = 'http://172.30.132.28:3001';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (!imagePath) {
      setError('Error: Please enter a valid image path.');
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${backendUrl}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_path: imagePath }),
      });

      if (!res.ok) {
        throw new Error('File not found or invalid file format. Try another image.');
      }

      const data = await res.json();
      setResponse(data);
      setDetectedImage(`${backendUrl}/output/${data.image_path}`);  // Display processed image
      
      const totalConfidence = data.detections.reduce((acc, detection) => acc + detection.confidence, 0);
      const avgConfidence = totalConfidence / data.detections.length *100;
      setAverageConfidence(avgConfidence);  // Set the average confidence state
    } catch (err) {
      setError('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Image Detection</h1>
      <form onSubmit={handleSubmit}>
        <label>
          <input
            type="text"
            value={imagePath}
            onChange={(e) => setImagePath(e.target.value)}
            placeholder="Enter image path"
          />
        </label>
        {/* Display 'Processing...' when loading */}
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Detect'} 
        </button> 
      </form>
      {/* //Display error message in red (mainly backend errors) */}
      {error && <p style={{ color: 'red' }}>{error}</p>} 

      {response && (
        <div>
          <h2>Detection Results</h2>
          <p><strong>Processed Image Path:</strong> {response.image_path}</p>
          <p><strong>Results File:</strong> {response.results_file}</p>
          <h3>Detected Objects:</h3>
          {/* <ul>
            {response.detections.map((detection, index) => (
              <li key={index}>
                Class ID: {detection.class_id}, Confidence: {detection.confidence}, Bounding Box: {detection.bounding_box.join(', ')}
              </li>
            ))}
          </ul> */}
          {averageConfidence && (
            <p><strong>Confidence:</strong> {averageConfidence.toFixed(4)} %</p>
          )}
          <h3>Processed Image:</h3>
          <img src={detectedImage} alt="Processed" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
      )}
    </div>
  );
}

export default App;
