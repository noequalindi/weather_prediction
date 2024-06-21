const isLocalhost = Boolean(
    window.location.hostname === 'localhost' ||
      // [::1] es la dirección IPv6 localhost.
      window.location.hostname === '[::1]' ||
      // 127.0.0.0/8 son consideradas direcciones IPv4 localhost.
      window.location.hostname.match(
        /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/
      )
  );
  
  export function register(config) {
    if (process.env.NODE_ENV === 'production' && 'serviceWorker' in navigator) {
      const publicUrl = new URL(process.env.PUBLIC_URL, window.location.href);
      if (publicUrl.origin !== window.location.origin) {
        return;
      }
  
      window.addEventListener('load', async () => {
        const swUrl = `${process.env.PUBLIC_URL}/service-worker.js`;
  
        if (isLocalhost) {
          // Está corriendo en localhost, verifica si existe un service worker registrado.
          checkValidServiceWorker(swUrl, config);
        } else {
          // No es localhost. Registra el service worker.
          registerValidSW(swUrl, config);
        }
      });
    }
  }
  
  async function registerValidSW(swUrl, config) {
    try {
      const registration = await navigator.serviceWorker.register(swUrl);
      registration.onupdatefound = () => {
        const installingWorker = registration.installing;
        if (installingWorker == null) {
          return;
        }
        installingWorker.onstatechange = () => {
          if (installingWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              console.log(
                'New content is available and will be used when all tabs for this page are closed.'
              );
              if (config && config.onUpdate) {
                config.onUpdate(registration);
              }
            } else {
              console.log('Content is cached for offline use.');
              if (config && config.onSuccess) {
                config.onSuccess(registration);
              }
            }
          }
        };
      };
    } catch (error) {
      console.error('Error during service worker registration:', error);
    }
  }
  
  function checkValidServiceWorker(swUrl, config) {
    fetch(swUrl)
      .then(async (response) => {
        if (
          response.status === 404 ||
          (response.headers.get('content-type')?.indexOf('javascript') === -1)
        ) {
          const registration = await navigator.serviceWorker.ready;
          registration.unregister().then(() => {
            window.location.reload();
          });
        } else {
          registerValidSW(swUrl, config);
        }
      })
      .catch(() => {
        console.log('No internet connection found. App is running in offline mode.');
      });
  }
  
  export function unregister() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready
        .then(async (registration) => {
          await registration.unregister();
        })
        .catch((error) => {
          console.error(error.message);
        });
    }
  }
  