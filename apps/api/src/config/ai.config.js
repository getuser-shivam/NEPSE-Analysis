/**
 * AI Services Configuration
 * 
 * Configuration for Groq AI and Pollens AI APIs.
 * API keys should be set in environment variables.
 */

import dotenv from 'dotenv';

dotenv.config();

export const aiConfig = {
  // Groq AI Configuration
  groq: {
    apiKey: process.env.GROQ_API_KEY || '',
    baseUrl: process.env.GROQ_BASE_URL || 'https://api.groq.com/openai/v1',
    model: process.env.GROQ_MODEL || 'llama-3.3-70b-versatile',
    maxTokens: parseInt(process.env.GROQ_MAX_TOKENS || '500', 10),
    temperature: parseFloat(process.env.GROQ_TEMPERATURE || '0.3'),
    timeout: parseInt(process.env.GROQ_TIMEOUT || '30000', 10),
  },

  // Pollens AI Configuration
  pollens: {
    apiKey: process.env.POLLENS_API_KEY || '', // Optional — Pollinations.ai is free
    baseUrl: process.env.POLLENS_BASE_URL || 'https://text.pollinations.ai',
    timeout: parseInt(process.env.POLLENS_TIMEOUT || '30000', 10),
  },

  // Feature flags
  features: {
    enableGroq: process.env.ENABLE_GROQ_AI === 'true' || false,
    enablePollens: process.env.ENABLE_POLLENS_AI === 'true' || false,
    fallbackEnabled: process.env.AI_FALLBACK_ENABLED !== 'false', // default true
  },

  // Rate limiting
  rateLimit: {
    maxRequestsPerMinute: parseInt(process.env.AI_RATE_LIMIT_PER_MINUTE || '10', 10),
    maxRequestsPerHour: parseInt(process.env.AI_RATE_LIMIT_PER_HOUR || '100', 10),
  },
};

// Validation
export function validateAIConfig() {
  const errors = [];

  if (aiConfig.features.enableGroq && !aiConfig.groq.apiKey) {
    errors.push('GROQ_API_KEY is required when Groq AI is enabled');
  }

  if (aiConfig.features.enablePollens && !aiConfig.pollens.apiKey) {
    errors.push('POLLENS_API_KEY is required when Pollens AI is enabled');
  }

  if (errors.length > 0) {
    console.warn('AI Configuration Warnings:', errors);
  }

  return errors.length === 0;
}

export default aiConfig;
