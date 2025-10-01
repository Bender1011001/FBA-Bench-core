(function() {
  if (typeof window !== 'undefined') {
    window.ANALYTICS_CONFIG = {
      enabled: false,
      provider: "placeholder",           // e.g., "plausible" or "custom"
      trackingId: "",                    // empty by default
      respectDNT: true,                  // respect Do Not Track
      sampleRate: 1.0,                   // 0–1.0; keep 1.0 default
      endpoint: ""                       // optional local or external endpoint (empty by default)
    };
  }
})();