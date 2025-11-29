import React, { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import ResultDisplay from './components/ResultDisplay'
import './styles/App.css'

function App() {
  const [personImage, setPersonImage] = useState(null)
  const [garmentImage, setGarmentImage] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleTryOn = async () => {
    if (!personImage || !garmentImage) {
      setError('Please upload both person and garment images')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('person_image', personImage)
      formData.append('garment_image', garmentImage)

      const response = await fetch('/api/v1/tryon', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const blob = await response.blob()
      const resultUrl = URL.createObjectURL(blob)
      setResult(resultUrl)
    } catch (err) {
      setError(`Try-on failed: ${err.message}`)
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setPersonImage(null)
    setGarmentImage(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Vesaki-VTON</h1>
        <p className="subtitle">High-Resolution Virtual Try-On System</p>
      </header>

      <main className="app-main">
        {!result ? (
          <div className="upload-section">
            <div className="upload-grid">
              <ImageUpload
                title="Person Image"
                description="Upload a photo of yourself"
                onImageSelect={setPersonImage}
                image={personImage}
              />
              <ImageUpload
                title="Garment Image"
                description="Upload the garment to try on"
                onImageSelect={setGarmentImage}
                image={garmentImage}
              />
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            <div className="action-buttons">
              <button
                className="btn btn-primary"
                onClick={handleTryOn}
                disabled={!personImage || !garmentImage || loading}
              >
                {loading ? 'Processing...' : 'Try On'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </div>
        ) : (
          <ResultDisplay
            personImage={URL.createObjectURL(personImage)}
            garmentImage={URL.createObjectURL(garmentImage)}
            resultImage={result}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by Vesaki-VTON | High-Resolution Virtual Try-On</p>
      </footer>
    </div>
  )
}

export default App

