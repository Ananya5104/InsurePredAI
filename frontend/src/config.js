// API configuration
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Deployed backend URL (used for admin access in production)
const DEPLOYED_BACKEND_URL = import.meta.env.VITE_DEPLOYED_BACKEND_URL || 'https://insurepredai-backend.onrender.com';

// Function to determine if we're in a production environment
const isProduction = () => {
  return import.meta.env.MODE === 'production' ||
         window.location.hostname !== 'localhost' &&
         window.location.hostname !== '127.0.0.1';
};

// Get the appropriate backend URL based on environment
const getBackendUrl = () => {
  return isProduction() ? DEPLOYED_BACKEND_URL : API_URL;
};

export { API_URL, DEPLOYED_BACKEND_URL, isProduction, getBackendUrl };
