import React, { useState } from 'react';
import { getBackendUrl } from '../config';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const RetrainModel = () => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRedirectToAdmin = () => {
    setLoading(true);

    // Get the appropriate backend URL based on environment
    const backendUrl = getBackendUrl();

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
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Retrain Churn Model</h2>

        <div className="space-y-6">
          <p className="text-gray-200">
            To retrain the model, you need to log in to the admin panel.
            Only administrators can access this feature to ensure data security.
          </p>
          <p className="text-gray-200 mt-2">
            Click the button below to go to the admin login page.
          </p>
          <p className="text-gray-200 mt-2 text-sm">
            Having trouble accessing the admin panel? Try the{" "}
            <Link to="/admin-access" className="text-yellow-300 hover:underline">
              direct admin access
            </Link>{" "}
            page.
          </p>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRedirectToAdmin}
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
                Go to Admin Panel
              </>
            )}
          </motion.button>
        </div>

        {message && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-6 p-4 rounded-lg ${message.includes('error') ? 'bg-red-500/30 border border-red-400' : 'bg-green-500/30 border border-green-400'}`}
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

export default RetrainModel;
