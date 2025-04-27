import React, { useState } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const RetrainModel = () => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRetrain = async () => {
    setLoading(true);
    setMessage('');

    try {
      const response = await axios.post(`${API_URL}/api/retrain-model/`);
      setMessage(response.data.message || 'Model retrained successfully!');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
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
            Retraining the model will use all the accumulated data to improve prediction accuracy.
            This process may take a few minutes to complete.
          </p>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRetrain}
            disabled={loading}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center"
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Retraining...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Retrain Model
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
