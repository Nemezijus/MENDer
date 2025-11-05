// MENDer/frontend/src/api/client.js
import axios from 'axios';

// Backend is running on 127.0.0.1:8000 from uvicorn
// const api = axios.create({
//   baseURL: 'http://127.0.0.1:8000/api/v1',
//   headers: {
//     'Content-Type': 'application/json',
//   },
// });

// export default api;

const api = axios.create({
  baseURL: '/api/v1',   // same for dev & prod
  timeout: 20000,
});

export default api;