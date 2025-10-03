(function() {
  'use strict';

  // Global configuration for FBA Bench site
  // Development defaults are intentionally safe and analytics-disabled.
  // Values can be overridden at build time via scripts/inject-config.sh using the @inject markers below.

  if (typeof window !== 'undefined') {
    window.FBABenchConfig = {
      analytics: {
        enabled: false, // @inject ANALYTICS_ENABLED (bool)
        trackingId: '', // @inject ANALYTICS_TRACKING_ID (string)
        endpoint: '', // @inject ANALYTICS_ENDPOINT (string URL)
        respectDNT: true // @inject ANALYTICS_RESPECT_DNT (bool)
      },
      environment: 'development' // @inject ENVIRONMENT (string)
    };
  }
})();