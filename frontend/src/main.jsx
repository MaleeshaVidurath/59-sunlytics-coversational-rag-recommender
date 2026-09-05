import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import App from './App.jsx';
import ErrorBoundary from './components/organisms/ErrorBoundary';
import { store } from './store';
import './styles/global.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <ErrorBoundary>
    <Provider store={store}>
      <App />
    </Provider>
  </ErrorBoundary>
);
