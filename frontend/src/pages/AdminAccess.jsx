import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { getBackendUrl } from '../config';

const AdminAccess = () => {
  const [backendUrl, setBackendUrl] = useState(getBackendUrl());
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleUrlChange = (e) => {
    setBackendUrl(e.target.value);
  };

  const handleAdminAccess = () => {
    if (!backendUrl) {
      setMessage('Please enter a valid backend URL');
      return;
    }

    setLoading(true);
    
    // Construct the admin URL
    const adminUrl = `${backendUrl}/admin/`;
    
    console.log(`Redirecting to admin panel at: ${adminUrl}`);
    
    // Redirect to the Django admin login page
    window.location.href = adminUrl;
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-black p-4"
    >
      {/* Back Button */}
      <div className="absolute top-4 left-4">
        <Link
          to="/"
          className="flex items-center text-white hover:text-yellow-300 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
          </svg>
          Back to Home
        </Link>
      </div>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="bg-white/10 backdrop-blur-lg p-8 rounded-2xl shadow-2xl max-w-md w-full border border-white/20"
      >
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Direct Admin Access</h2>

        <div className="space-y-6">
          <p className="text-gray-200">
            If you're having trouble accessing the admin panel, you can use this direct access form.
            Enter the backend URL and click the button below.
          </p>

          <div className="space-y-4">
            <div>
              <label htmlFor="backendUrl" className="block text-sm font-medium text-gray-300 mb-1">
                Backend URL
              </label>
              <input
                type="text"
                id="backendUrl"
                value={backendUrl}
                onChange={handleUrlChange}
                className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                placeholder="https://your-backend-url.com"
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleAdminAccess}
              disabled={loading}
              className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Redirecting...
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Access Admin Panel
                </>
              )}
            </motion.button>
          </div>

          <div className="mt-4">
            <p className="text-gray-300 text-sm">
              Common backend URLs:
            </p>
            <ul className="text-gray-300 text-sm list-disc pl-5 mt-2">
              <li>Local: http://127.0.0.1:8000</li>
              <li>Deployed: https://insurepredai-backend.onrender.com</li>
            </ul>
          </div>
        </div>

        {message && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 rounded-lg bg-red-500/30 border border-red-400"
          >
            <p className="text-white font-medium">
              {message}
            </p>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default AdminAccess;
