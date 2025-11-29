import React, { useRef } from 'react'
import '../styles/ImageUpload.css'

function ImageUpload({ title, description, onImageSelect, image }) {
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      onImageSelect(file)
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="image-upload">
      <h3>{title}</h3>
      <p className="description">{description}</p>

      <div className="upload-area" onClick={handleClick}>
        {image ? (
          <div className="preview">
            <img src={URL.createObjectURL(image)} alt="Preview" />
            <div className="overlay">
              <span>Click to change</span>
            </div>
          </div>
        ) : (
          <div className="placeholder">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <p>Click to upload</p>
            <span className="hint">JPG, PNG up to 10MB</span>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
    </div>
  )
}

export default ImageUpload

