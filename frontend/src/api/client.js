import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 0,
});

export default api;