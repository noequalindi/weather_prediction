// service-worker.js

import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';

// Precache todos los assets definidos en el manifest.
// eslint-disable-next-line no-restricted-globals
precacheAndRoute(self.__WB_MANIFEST);

// Configura las estrategias de cache para las rutas.

// Ruta para las peticiones al backend (ajusta esta URL según tu configuración).
registerRoute(
  ({url}) => url.pathname.startsWith('/api'),
  new NetworkFirst()
);

// Cache para los assets de terceros (por ejemplo, CDN).
registerRoute(
  ({url}) => url.origin.startsWith('https://cdn.example.com'),
  new StaleWhileRevalidate()
);

// Cualquier otro recurso usa StaleWhileRevalidate.
registerRoute(
  ({url}) => true,
  new StaleWhileRevalidate()
);
