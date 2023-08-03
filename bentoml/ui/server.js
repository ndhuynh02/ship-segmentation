import express from "express";
import multer from "multer";
import path from "path";
import fetch from "node-fetch";
import FormData from "form-data"

// const express = require('express');
// const multer = require('multer');
// const path = require('path');
// const fetch = require('node-fetch');
// const FormData = require('form-data'); 
const app = express();
const port = 4000;

// Set up storage for uploaded images using Multer with memory storage
const upload = multer({ storage: multer.memoryStorage() });

// Enable CORS (optional if you've already configured this)
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Route to handle POST request containing an image
app.post('/segment', upload.single('image'), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: 'No image file provided.' });
  } else {
    const processedImage = await processImage(req.file.buffer);
    if (!processedImage) {
      res.status(500).json({ error: 'Failed to process the image.' });
    } else {
      res.contentType('image/jpeg'); // Set the content type to indicate it's an image
      res.end(processedImage); // Send the processed image as a binary response
    }
  }
});

// Function to process the uploaded image using BentoML API
async function processImage(imageBuffer) {
  try {
    // host.docker.internal:3000 if using docker
    // localhost:3000 otherwise
    const apiEndpoint = 'http://host.docker.internal:3000/segment'; // Replace with your BentoML API endpoint
    const formData = new FormData();
    formData.append('image', Buffer.from(imageBuffer), { filename: 'image.jpg' });

    const response = await fetch(apiEndpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('BentoML API response was not ok');
    }

    return response.buffer(); // Return the image response as a buffer
  } catch (error) {
    console.error('Error:', error);
    return null;
  }
}

// Start the server
app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
