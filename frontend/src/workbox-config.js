module.exports = {
  "globDirectory": "build/",
  "globPatterns": [
    "**/*.{json,ico,html,js,css}"
  ],
  "swDest": "build/service-worker.js",
  "clientsClaim": true,
  "skipWaiting": true,
  "runtimeCaching": [
    {
      "urlPattern": /\.(?:png|jpg|jpeg|svg|gif)$/,
      "handler": "CacheFirst",
      "options": {
        "cacheName": "images-cache",
        "expiration": {
          "maxEntries": 50
        }
      }
    }
  ],
  "ignoreURLParametersMatching": [
    /^utm_/,
    /^source/,
    /^version/
  ]
};
