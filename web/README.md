# Vesaki-VTON Web Interface

Modern web interface for the Vesaki-VTON virtual try-on system.

## Features

- Drag-and-drop image upload
- Real-time try-on processing
- Side-by-side result comparison
- Download results
- Responsive design (mobile-friendly)
- Modern gradient UI

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Server will run on `http://localhost:3000`

## Build for Production

```bash
# Build optimized bundle
npm run build

# Preview production build
npm run preview
```

## API Configuration

The web app connects to the Vesaki-VTON API at `http://localhost:8000`.

To change the API endpoint, edit `vite.config.js`:

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://your-api-url:8000',
      changeOrigin: true
    }
  }
}
```

## Deployment

### Static Hosting (Netlify, Vercel)

```bash
npm run build
# Upload dist/ folder
```

### Docker

```bash
docker build -t vesaki-vton-web -f Dockerfile.web .
docker run -p 3000:3000 vesaki-vton-web
```

### With Backend

Use Docker Compose from project root:
```bash
docker-compose up -d
```

Web interface will be available at `http://localhost`

## Technology Stack

- React 18
- Vite (build tool)
- Axios (HTTP client)
- Modern CSS with gradients

---

**Part of Vesaki-VTON System**

