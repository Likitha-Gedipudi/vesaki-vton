import React from 'react'
import '../styles/ResultDisplay.css'

function ResultDisplay({ personImage, garmentImage, resultImage, onReset }) {
  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = resultImage
    link.download = 'vesaki-vton-result.jpg'
    link.click()
  }

  return (
    <div className="result-display">
      <h2>Try-On Result</h2>

      <div className="result-grid">
        <div className="result-item">
          <div className="result-label">Original</div>
          <img src={personImage} alt="Original person" />
        </div>

        <div className="result-item">
          <div className="result-label">Garment</div>
          <img src={garmentImage} alt="Garment" />
        </div>

        <div className="result-item result-highlight">
          <div className="result-label">Result</div>
          <img src={resultImage} alt="Try-on result" />
        </div>
      </div>

      <div className="result-actions">
        <button className="btn btn-primary" onClick={handleDownload}>
          Download Result
        </button>
        <button className="btn btn-secondary" onClick={onReset}>
          Try Another
        </button>
      </div>
    </div>
  )
}

export default ResultDisplay

