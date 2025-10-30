import React from 'react';
import { createRoot } from 'react-dom/client';
import { MantineProvider } from '@mantine/core';
import '@mantine/core/styles.css';
import App from './App.jsx';

const rootEl = document.getElementById('root');

createRoot(rootEl).render(
  <React.StrictMode>
    <MantineProvider
      defaultColorScheme="light"
      // you can also pass theme overrides here later if you want
    >
      <App />
    </MantineProvider>
  </React.StrictMode>
);
