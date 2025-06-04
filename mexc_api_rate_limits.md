# MEXC API Rate Limits Documentation

## Overview

This document summarizes the rate limits for the MEXC API based on official documentation and announcements. Proper adherence to these rate limits is essential for maintaining stable connectivity and avoiding request rejections.

## General Rate Limit Information

When API requests exceed the defined rate limits, the MEXC API will return:
- HTTP Status Code: `429`
- Error Message: `Too many requests`

## Spot Order Rate Limits

As of **March 25, 2025**, MEXC has adjusted the API Spot order limit:

- **Spot Orders**: 5 orders per second
- **Batch Orders**: This limit is also applicable to batch orders (each batch counts as one request, but has its own internal limit)

Source: [Official MEXC Announcement (March 22, 2025)](https://www.mexc.com/support/articles/17827791522801)

## Batch Orders Limit

- **Batch Size**: Supports up to 20 orders in a single batch
- **Rate Limit**: 2 times per second

Source: MEXC API Documentation (Change Log July 8, 2022)

## Other Endpoint-Specific Rate Limits

While not all endpoints have explicitly documented rate limits, the following information has been gathered from the API documentation:

| Endpoint | Rate Limit | Notes |
|----------|------------|-------|
| Spot Orders | 5 orders/second | Effective March 25, 2025 |
| Batch Orders | 2 requests/second | Each batch can contain up to 20 orders |
| Contract Support Currencies | 20 times/2 seconds | As mentioned in documentation |

## Best Practices for Rate Limit Compliance

1. **Implement Request Spacing**:
   - Add delays between requests to ensure compliance with rate limits
   - For spot orders, ensure no more than 5 orders are sent per second

2. **Use Exponential Backoff**:
   - When receiving a 429 error, implement exponential backoff before retrying
   - Start with a base delay (e.g., 1 second) and increase exponentially with each retry

3. **Batch Requests When Possible**:
   - Use batch order endpoints for multiple orders
   - Remember that batch requests have their own rate limits (2/second)

4. **Prioritize Critical Requests**:
   - When approaching rate limits, prioritize critical operations (e.g., order execution) over informational requests

5. **Implement Request Queuing**:
   - Queue requests and process them at a controlled rate to avoid exceeding limits
   - Consider using a token bucket algorithm for rate limiting

## Implementation Recommendations for Trading-Agent System

For the Trading-Agent system focusing on BTCUSDC:

1. **Market Data Collection**:
   - Implement a minimum interval of 200ms between market data requests
   - Use websocket connections where possible to reduce REST API calls

2. **Order Execution**:
   - Ensure no more than 5 order operations per second
   - Implement a queue system for orders if multiple signals are generated simultaneously

3. **Error Handling**:
   - Monitor for 429 responses
   - Implement automatic retry with exponential backoff
   - Log all rate limit incidents for analysis

## References

1. [MEXC to Adjust API Spot Order Rate Limit (March 22, 2025)](https://www.mexc.com/support/articles/17827791522801)
2. [MEXC API Documentation - Introduction](https://mexcdevelop.github.io/apidocs/spot_v3_en/)
3. [MEXC API Documentation - Change Log](https://mexcdevelop.github.io/apidocs/spot_v3_en/#change-log)
